"""Official Dream YAM policy with a persistent budget and one bounded session recovery."""
import time
import warnings

from dream_budget import BudgetStopped, Ledger


class RequestStopped(RuntimeError):
    pass


def guard_exchange(original, ledger, task, run, stats=None):
    """Wrap the actual transport exchange, below SDK connection retry logic."""
    def exchange(transport, body, headers, path="/v1/get_action", timeout_ms=None):
        if path == "/v1/end_session":
            return original(transport, body, headers, path, timeout_ms)
        if path != "/v1/get_action":
            raise RequestStopped("unexpected_api_route")
        reservation = ledger.reserve(task, run)
        if stats is not None: stats["reserved_calls"] += 1
        print(f"[dream] request reserved id={reservation} rate_ceiling_usd=0.0015", flush=True)
        try:
            # Rendering 48 rows can outlive a proxy's keep-alive timeout. Start a fresh
            # TLS connection for each chunk, without retrying any inference request.
            transport._drop()
            reply = original(transport, body, headers, path, timeout_ms)
        except Exception as error:
            ledger.finish(reservation, "network_error")
            # Includes stale-connection exceptions: do not let the SDK silently retry them.
            raise RequestStopped("network_request_failed:" + type(error).__name__) from None
        ledger.finish(reservation, "response")
        return reply
    return exchange


def request_with_recovery(client, commanded, expired_type, stats):
    """At most one session replacement per rollout; preserve observations and simulator state."""
    while True:
        try:
            return client.get_action(commanded=commanded)
        except expired_type:
            if stats['session_recoveries'] >= 1:
                raise
            stats['session_recoveries'] += 1
            print('[dream] Session replaced; continuing current scene once, subject to the same budget',flush=True)


class Runner:
    def __init__(self, task, run, instruction, budget_path, seed=0):
        import dream
        import dream.transport
        from inspect_robots import EmbodimentInfo, Scene
        from inspect_robots_dream import _station
        from inspect_robots_dream.policy import DreamPolicy
        from inspect_robots_dream.session import DreamSession, Chunk

        if dream.__version__ != "1.0.14":
            raise RequestStopped("Dream SDK changed; review transport guard before spending")
        self.ledger = Ledger(budget_path)
        self.task, self.run = task, run
        self.stats = {"reserved_calls": 0, "session_recoveries": 0}
        self.ledger.status(task, run)  # Fail before opening a client if the ledger is missing.
        self._original = dream.transport.Transport._exchange
        dream.transport.Transport._exchange = guard_exchange(self._original, self.ledger, task, run, self.stats)
        warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)

        class SingleAttemptSession(DreamSession):
            def act(session, commanded):
                client = session._open()
                sent = time.monotonic()
                recovered_before=self.stats["session_recoveries"]
                try:
                    reply = request_with_recovery(client, None if commanded is None else _station.commanded(commanded), dream.DreamSessionExpired, self.stats)
                except (BudgetStopped, RequestStopped):
                    raise
                except Exception as error:
                    # Log only an error class and its documented reason, never request headers/key.
                    reason = getattr(error, "reason", None)
                    if reason not in {"over_cap", "not_entitled", "missing", "unknown_key", "disabled",
                                      "pool_warming", "pool_full", "cannot_scale", "client_outdated",
                                      "expired", "evicted", "restarted", "replaced"}:
                        reason = "request_refused"
                    raise RequestStopped(f"{type(error).__name__}:{reason}") from None
                timing = client.last_timing or {}
                return Chunk(_station.rows(reply), time.monotonic()-sent, 1+self.stats["session_recoveries"]-recovered_before,
                             timing.get("serve_infer_ms"), client.session_id)

        self.policy = DreamPolicy(session=SingleAttemptSession(), record=False)
        try:
            self.policy.bind(EmbodimentInfo(name="bimanual_yam", action_space=self.policy.info.action_space,
                observation_space=self.policy.info.observation_space, control_hz=20, is_simulated=True))
            self.policy.reset(Scene(id=task, instruction=instruction, init_seed=seed))
        except Exception:
            self.close()
            raise

    def act(self, images, state):
        from inspect_robots import Observation
        return self.policy.act(Observation(images=images, state={"joint_pos": state})).actions[0].data

    def report(self):
        try:
            budget = self.ledger.status(self.task, self.run)
        except BudgetStopped as error:
            budget = {"error": str(error), "reserved": {"run_calls": self.stats["reserved_calls"]}}
        return {"budget": budget, "policy": self.policy.transcript(), "session_recoveries": self.stats["session_recoveries"]}

    def close(self):
        import dream.transport
        try:
            self.policy.close()
        finally:
            dream.transport.Transport._exchange = self._original
