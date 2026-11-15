import hashlib
import json

import pytest

from rcauto.benchmarks import collect_repository, select_definitions, validate_benchmarks
from rcauto.gates import GateError, file_hash
from rcauto.store import write_json


def datacenter_sources():
    setups = {
        "tasks": [
            {
                "id": f"DC-{i}",
                "design": f"design-{i}",
                "name": f"Task {i}",
                "setup": "A reproducible physical setup.",
                "reset": "Restore the initial state.",
                "instruction": "Perform the physical operation.",
                "levels": [
                    {"score": 0, "description": "Unchanged."},
                    {"score": 1, "description": "The goal was reached."},
                ],
            }
            for i in range(10)
        ]
    }
    designs = {
        "groups": [
            {
                "tasks": [
                    {"id": f"design-{i}", "real": {"procedure": "Manual step."}} for i in range(19)
                ]
            }
        ]
    }
    return setups, designs


def test_datacenter_collection_retains_all_ten_original_rubrics(tmp_path):
    from rcauto.benchmarks import collect_datacenter_bench

    setups, designs = datacenter_sources()
    calls = []

    def fetch(url, destination):
        calls.append(url)
        raw = json.dumps(setups if url.endswith("setups.json") else designs)
        return {"url": url, "text": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}

    files, catalog = collect_datacenter_bench(tmp_path / "definitions", tmp_path / "cache", fetch)
    assert len(files) == 2 and catalog["task_count"] == 10
    assert len(catalog["selected_design_ids"]) == 10
    original = json.loads((tmp_path / "definitions/official-setups.json").read_text())
    assert original == setups
    assert all(t["levels"][-1]["score"] == 1 for t in original["tasks"])
    assert all("/data/" in url for url in calls)
    corpus = tmp_path / "input/manifest.json"
    write_json(
        corpus,
        {
            "complete": True,
            "task_catalog": catalog,
            "files": [{**item, "local_path": "definitions/" + item["path"]} for item in files],
        },
    )
    write_json(
        tmp_path / "input/benchmark-sources.json",
        {
            "DataCenterBench": {
                "complete": True,
                "definition_manifest": "input/manifest.json",
                "definition_files": 2,
                "manifest_sha256": file_hash(corpus),
            }
        },
    )
    validate_benchmarks(tmp_path, ["DataCenterBench"])


@pytest.mark.parametrize("failure", ["missing-rubric", "missing-task", "unlinked-design"])
def test_datacenter_collection_refuses_incomplete_definitions(tmp_path, failure):
    from rcauto.benchmarks import collect_datacenter_bench

    setups, designs = datacenter_sources()
    if failure == "missing-rubric":
        del setups["tasks"][0]["levels"]
    elif failure == "missing-task":
        setups["tasks"].pop()
    else:
        setups["tasks"][0]["design"] = "unavailable"

    def fetch(url, destination):
        raw = json.dumps(setups if url.endswith("setups.json") else designs)
        return {"url": url, "text": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}

    with pytest.raises(GateError, match="DataCenterBench"):
        collect_datacenter_bench(tmp_path / "definitions", tmp_path / "cache", fetch)


def test_datacenter_overview_snapshot_is_not_complete_original_coverage(tmp_path):
    source = tmp_path / "input/overview.txt"
    source.parent.mkdir(parents=True)
    source.write_text("An overview listing ten task names, without their scoring rubrics.")
    corpus = tmp_path / "input/manifest.json"
    write_json(
        corpus,
        {
            "complete": True,
            "files": [{"local_path": "input/overview.txt", "sha256": file_hash(source)}],
        },
    )
    write_json(
        tmp_path / "input/benchmark-sources.json",
        {
            "DataCenterBench": {
                "complete": True,
                "definition_manifest": "input/manifest.json",
                "definition_files": 1,
                "manifest_sha256": file_hash(corpus),
            }
        },
    )
    with pytest.raises(GateError, match="DataCenterBench"):
        validate_benchmarks(tmp_path, ["DataCenterBench"])


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def test_definition_selection_rejects_truncated_trees_empty_groups_and_unsafe_paths():
    entry = {"path": "tasks/nested/task.py", "type": "blob", "mode": "100644", "size": 10}
    assert select_definitions({"truncated": False, "tree": [entry]}, ["tasks/*.py"]) == [entry]
    for tree, patterns in [
        ({"truncated": True, "tree": [entry]}, ["tasks/*.py"]),
        ({"truncated": False, "tree": [entry]}, ["definitions/*.bddl"]),
        (
            {"truncated": False, "tree": [{**entry, "path": "tasks/../../escape.py"}]},
            ["tasks/*.py"],
        ),
        ({"truncated": False, "tree": [{**entry, "mode": "120000"}]}, ["tasks/*.py"]),
    ]:
        with pytest.raises(GateError):
            select_definitions(tree, patterns)


@pytest.mark.parametrize("raw", [b'(:language "turn the knob")\n(:goal (turned knob))\n', b""])
def test_corpus_verifies_pinned_blob_bytes_and_reuses_only_valid_cache(tmp_path, raw):
    entry = {
        "path": "tasks/a.bddl",
        "type": "blob",
        "mode": "100644",
        "size": len(raw),
        "sha": blob(raw),
    }
    tree = {"truncated": False, "sha": "b" * 40, "tree": [entry]}

    class API:
        def get(self, endpoint, **kwargs):
            if endpoint.endswith("/commits/HEAD"):
                return {"sha": "a" * 40, "commit": {"tree": {"sha": "b" * 40}}}
            return tree

    calls = []

    def fetch(url, expected_size):
        calls.append(url)
        return raw

    cache = tmp_path / "cache"
    result = collect_repository("owner/repo", ["tasks/*.bddl"], cache, API(), fetch=fetch)
    assert result["complete"] and len(result["files"]) == 1
    item = result["files"][0]
    assert (
        item["url"] == "https://raw.githubusercontent.com/owner/repo/" + "a" * 40 + "/tasks/a.bddl"
    )
    assert item["sha256"] == hashlib.sha256(raw).hexdigest()
    collect_repository("owner/repo", ["tasks/*.bddl"], cache, API(), fetch=fetch)
    assert len(calls) == 1
    (cache / item["cache_path"]).write_bytes(b"corrupt")
    collect_repository("owner/repo", ["tasks/*.bddl"], cache, API(), fetch=fetch)
    assert len(calls) == 2
    (cache / item["cache_path"]).unlink()
    with pytest.raises(GateError, match="blob"):
        collect_repository(
            "owner/repo", ["tasks/*.bddl"], cache, API(), fetch=lambda *a: b"wrong HTTP 200 content"
        )


def test_overview_is_not_a_definition_corpus_and_changed_original_cannot_pass(tmp_path):
    with pytest.raises(GateError, match="definitions"):
        validate_benchmarks(tmp_path, ["LIBERO"])
    write_json(tmp_path / "input/benchmark-sources.json", {"LIBERO": {"text": "overview"}})
    with pytest.raises(GateError, match="definitions"):
        validate_benchmarks(tmp_path, ["LIBERO"])
    source = tmp_path / "input/task.bddl"
    source.write_text("(:goal (done task))")
    corpus = tmp_path / "input/manifest.json"
    write_json(
        corpus,
        {
            "complete": True,
            "files": [
                {
                    "local_path": "input/task.bddl",
                    "sha256": file_hash(source),
                }
            ],
        },
    )
    write_json(
        tmp_path / "input/benchmark-sources.json",
        {
            "LIBERO": {
                "complete": True,
                "definition_manifest": "input/manifest.json",
                "definition_files": 1,
                "manifest_sha256": file_hash(corpus),
            }
        },
    )
    validate_benchmarks(tmp_path, ["LIBERO"])
    source.write_text("changed goal")
    with pytest.raises(GateError, match="changed"):
        validate_benchmarks(tmp_path, ["LIBERO"])


def test_missing_originals_stop_before_expensive_catalog_review(tmp_path):
    from rcauto.engine import Production
    from rcauto.store import Store

    store = Store(tmp_path / "state")
    run = store.create("industrial task", instance="test")
    production = Production(store, run["id"], None, {"benchmarks": ["LIBERO"]})
    with pytest.raises(GateError, match="definitions"):
        production.novelty()
