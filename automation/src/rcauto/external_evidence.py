"""Honor explicit storage decisions for failed diagnostics, never release inputs."""

import json
import re

from .gates import GateError, file_hash, inside
from .store import write_json


def _verified(path, sha, size=None):
    if (path.is_symlink() or not path.is_file() or file_hash(path) != sha
            or (size is not None and path.stat().st_size != size)):
        raise GateError("external diagnostic evidence is missing or changed: " + path.name)


def externalize_failed_diagnostics(run_root, public_root):
    """Remove only verified copies; original/CAS/private backup bytes remain intact."""
    policy_path = run_root / "external-evidence-policy.json"
    if not policy_path.exists():
        return []
    try:
        policy = json.loads(policy_path.read_text())
        if (policy.get("scope") != "failed-native-diagnostics"
                or not policy.get("actor") or not policy.get("statement")
                or not isinstance(policy.get("files"), list) or not policy["files"]):
            raise GateError("explicit failed-diagnostic storage decision required")
        workspace = run_root / "workspace"
        omissions = []
        public_entries = []
        for entry in policy["files"]:
            sha, report_sha = entry["sha256"], entry["report_sha256"]
            if not all(isinstance(s, str) and re.fullmatch(r"[a-f0-9]{64}", s)
                       for s in (sha, report_sha)):
                raise GateError("invalid external diagnostic digest")
            size = entry["bytes"]
            if not isinstance(size, int) or size < 1024 * 1024:
                raise GateError("externalization is only for large diagnostic traces")
            blob = f"measurements/native-output-blobs/{sha}.json"
            _verified(inside(workspace, blob), sha, size)
            _verified(inside(public_root, blob), sha, size)
            _verified(inside(run_root / "private-experimental-evidence", sha + ".json"), sha, size)
            report_path = f"measurements/native-output-blobs/{report_sha}.json"
            report_file = inside(workspace, report_path)
            _verified(report_file, report_sha)
            report = json.loads(report_file.read_text())
            if (report.get("qualification") is not False
                    or report.get("overall_pass") is not False
                    or report.get("native_trial_id") != entry["trial"]):
                raise GateError("only a recorded failed native preflight can be externalized")
            summary = entry["summary"]
            if not re.fullmatch(r"measurements/[a-zA-Z0-9_.-]+-state-trace\.json", summary):
                raise GateError("public state trace required for external diagnostic")
            summary_file = inside(public_root, summary)
            _verified(summary_file, entry["summary_sha256"])
            provenance = json.loads(summary_file.read_text()).get("provenance", {})
            if (provenance.get("raw_trace_sha256") != sha
                    or provenance.get("qualification") is not False):
                raise GateError("public state trace does not identify the omitted raw evidence")
            paths = [blob]
            for alias in entry.get("aliases", []):
                if (not re.fullmatch(r"measurements/[a-zA-Z0-9_.-]+-trace\.json", alias)
                        or alias not in report.get("artifacts", [])):
                    raise GateError("only the failed report's raw trace alias may be omitted")
                source = inside(workspace, alias)
                # Later trials may overwrite the mutable alias. Never hide new bytes.
                if source.is_file() and file_hash(source) == sha:
                    _verified(inside(public_root, alias), sha, size)
                    paths.append(alias)
            omissions.extend(paths)
            public_entries.append({
                "sha256": sha, "bytes": size, "native_trial_id": entry["trial"],
                "failed_report": report_path, "omitted_paths": paths,
                "public_state_trace": summary, "state_trace_sha256": entry["summary_sha256"],
                "retained_locations": ["Mac", "Brev"],
            })
        # Validate the entire decision before modifying even the public copy.
        for name in set(omissions):
            inside(public_root, name).unlink()
        write_json(public_root / "external-evidence.json", {
            "full_raw_in_checkpoint": False,
            "scope": "Failed diagnostic originals retained on Mac and Brev. Full raw traces "
                     "cannot be restored from this GitHub checkpoint alone.",
            "files": public_entries,
        })
        return public_entries
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise GateError("invalid external diagnostic storage decision: " + str(error)) from error
