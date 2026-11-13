"""Offline control-flow fixture, explicitly ineligible for evaluation/approval/submission."""

import json
import sys
from pathlib import Path

from .engine import Production
from .store import write_json


class FixtureAgent:
    def run(self, role, prompt, workspace, schema, writable=False):
        workspace = Path(workspace)
        if role == "task-ideator":
            candidates = [
                {
                    "id": str(i),
                    "title": "FIXTURE ONLY",
                    "objective": "test coordinator",
                    "skills": ["fixture"],
                    "success_conditions": "fixture",
                    "industrial_value": "none: not an industrial task",
                    "dexterity": "none",
                    "evidence": [
                        {
                            "source_id": "fixture",
                            "quote": "fixture source text for control flow",
                            "locator": "fixture",
                        }
                    ],
                    "asset_plan": "fixture",
                    "risk": "none",
                }
                for i in range(3)
            ]
            return {
                "candidates": candidates,
                "selected_id": "0",
                "selection_reason": "fixture only",
            }
        if role == "novelty-reviewer":
            data = json.loads((workspace / "input/novelty-batch.json").read_text())
            return {
                "comparisons": [
                    {
                        "id": r["id"],
                        "objective": "fixture",
                        "physical_skill": "fixture",
                        "success_condition": "fixture",
                        "verdict": "distinct",
                        "reason": "fixture",
                    }
                    for r in data["records"]
                ]
            }
        if role == "task-specifier":
            return {
                "slug": "fixture-only",
                "objective": "fixture",
                "instruction": "fixture",
                "industrial_mapping": "not an industry task",
                "run_steps": 10,
                "physics_checks": [
                    {"name": f"fixture-{i}", "operator": "==", "threshold": 0, "unit": "fixture"}
                    for i in range(4)
                ],
                "damage_rules": ["fixture"],
                "assets": ["fixture"],
                "approximations": ["all synthetic"],
                "scoring_spec": "fixture",
                "required_controls": [
                    {"name": "fixture-control", "expected_score": 0, "reason": "fixture"}
                ],
            }
        if role in {"environment-builder", "demo-and-scoring-builder", "export-builder"}:
            if role == "environment-builder":
                requirements = json.loads((workspace / "artifacts/contract.json").read_text())[
                    "physics_checks"
                ]
                plan = self.scripts(workspace, {"environment": requirements})
                from .gates import file_hash

                (workspace / "task/LICENSE").write_text("Synthetic test fixture only.\n")
                write_json(
                    workspace / "task/asset-manifest.json",
                    {
                        "assets": [
                            {
                                "path": "environment.py",
                                "sha256": file_hash(workspace / "task/environment.py"),
                                "source_url": "https://example.com/fixture",
                                "license": "fixture",
                                "license_reference": "LICENSE",
                                "units": "m",
                                "dimensions_m": [0.1, 0.1, 0.1],
                                "geometry_kind": "procedural",
                            }
                        ]
                    },
                )
                return plan
            if role == "demo-and-scoring-builder":
                write_json(
                    workspace / "task/task.json",
                    {
                        "title": "FIXTURE ONLY",
                        "category": "Impact on AI timelines",
                        "instruction": "fixture",
                        "setup": "fixture",
                        "scoring": "fixture",
                        "run_limit": "10",
                        "fidelity": "synthetic test fixture, no robotics validation",
                        "scores": "fixture only",
                    },
                )
                (workspace / "task/task.py").write_text("# fixture: not a robot task\n")
                (workspace / "task/README.md").write_text(
                    "Offline fixture. Not an industrial task or submission.\n"
                )
                return self.scripts(
                    workspace,
                    {
                        "demo": [
                            {
                                "name": "full_score_episode_count",
                                "operator": ">=",
                                "threshold": 2,
                                "unit": "episodes",
                            },
                            {
                                "name": "object_pose_injections",
                                "operator": "==",
                                "threshold": 0,
                                "unit": "writes",
                            },
                        ],
                        "controls": [
                            {
                                "name": "wrong_score_count",
                                "operator": "==",
                                "threshold": 0,
                                "unit": "episodes",
                            }
                        ],
                        "replay": [
                            {
                                "name": "score_error",
                                "operator": "<=",
                                "threshold": 1e-6,
                                "unit": "score",
                            }
                        ],
                    },
                )
            write_json(
                workspace / "release-map.json",
                {
                    "demo_video": "measurements/fixture.mp4",
                    "screenshot": "measurements/setup.png",
                    "trace": "measurements/trace.json",
                    "episode_log": "measurements/episode.json",
                },
            )
            import subprocess

            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=blue:s=320x176:r=20",
                    "-t",
                    "0.2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                    str(workspace / "measurements/fixture.mp4"),
                ],
                check=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-i",
                    str(workspace / "measurements/fixture.mp4"),
                    "-frames:v",
                    "1",
                    "-y",
                    str(workspace / "measurements/setup.png"),
                ],
                check=True,
            )
            write_json(workspace / "measurements/trace.json", {"fixture": True})
            write_json(workspace / "measurements/episode.json", {"fixture": True})
            return self.scripts(
                workspace,
                {
                    "portable": [
                        {
                            "name": "fixture_copy",
                            "operator": "==",
                            "threshold": 0,
                            "unit": "fixture",
                        }
                    ]
                },
            )
        return {
            "verdict": "pass",
            "findings": ["synthetic fixture, not a scientific or robotics result"],
            "evidence_paths": ["input/request.json"],
        }

    def scripts(self, workspace, groups):
        scripts = []
        for name, requirements in groups.items():
            report = "measurements/" + name + ".json"
            checks = [dict(r, measured=r["threshold"]) for r in requirements]
            payload = {"checks": checks, "requirements": requirements, "artifacts": []}
            if name == "controls":
                payload["cases"] = [{"name": "fixture-control", "score": 0}]
            code = (
                "import json\nfrom pathlib import Path\n"
                "root=Path(__file__).resolve().parent.parent\n"
                f"report={payload!r}\n"
                f"(root/{report!r}).write_text(json.dumps(report))\n"
            )
            (workspace / "task" / (name + ".py")).write_text(code)
            scripts.append({"id": name, "script": name + ".py", "args": [], "report": report})
        return {"summary": "synthetic fixture", "scripts": scripts, "files": []}


class FixtureProduction(Production):
    def research(self):
        write_json(
            self.workspace / "input/sources.json",
            {"fixture": {"text": "fixture source text for control flow"}},
        )
        write_json(self.workspace / "input/benchmark-sources.json", {"fixture": True})
        return {"fixture": True}


def run(store):
    r = store.create("OFFLINE FIXTURE ONLY", instance="none", mode="selftest")
    config = {"runtime": {"python": sys.executable}, "repair_attempts": 0}
    pipeline = FixtureProduction(store, r["id"], FixtureAgent(), config)
    write_json(pipeline.workspace / "input/request.json", r)
    write_json(
        pipeline.workspace / "input/catalog.json",
        {"complete": True, "digest": "fixture", "records": [{"id": "fixture-1"}]},
    )
    pipeline.run()
    store.update(r["id"], status="selftest-complete", compute="not-used")
    return {
        "run": r["id"],
        "status": "selftest-complete",
        "stages": store.completed(r["id"]),
        "gpu_used": False,
        "model_calls": 0,
        "dream_calls": 0,
        "submissions": 0,
        "approval_allowed": False,
        "report": str(store.run_dir(r["id"])),
    }
