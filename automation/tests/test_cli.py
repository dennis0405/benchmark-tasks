from rcauto import cli
from rcauto.store import Store


def test_stage_only_resume_preserves_user_revision_constraints(tmp_path, monkeypatch):
    store = Store(tmp_path)
    run = store.create("industrial task", instance="vm")
    store.update(
        run["id"], revision=3, revision_request="Exclude connectors and sensor-clearance tasks"
    )
    monkeypatch.setattr(cli, "execute_or_detach", lambda args, state, *rest: state.get(run["id"]))
    args = cli.parser().parse_args(["resume", run["id"], "--from-stage", "research", "--detach"])
    result = cli.dispatch(args, store, {"instance": "vm", "instance_id": "id"})
    assert result["revision"] == 4
    assert result["revision_request"] == "Exclude connectors and sensor-clearance tasks"
    assert result["resume_from_stage"] == "research"
