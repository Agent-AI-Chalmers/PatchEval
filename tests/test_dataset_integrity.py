import json
import re
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[1] / "patcheval/datasets"


def load(name):
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def by_cve(name):
    return {entry["cve_id"]: entry for entry in load(name)}


DANGLING = re.compile(
    r"referenced (?:GitHub Security Advisory|GitHub Advisory|GHSA)(?![-\w])"
)
FIX_HINT = re.compile(r"before version|patched in|fixed in|upgrade to", re.IGNORECASE)


def test_descriptions_are_complete_after_in_place_fixes():
    entries = by_cve("patcheval_verified_cleaned.json")

    assert len(entries) == 230
    shescape = entries["CVE-2021-21384"]["cve_description"]
    assert "insert a null character into the payload" in shescape
    grafana = entries["CVE-2021-43798"]["cve_description"]
    assert "plugins/<plugin-id>/" in grafana
    assert "where is the plugin ID" not in grafana


def test_no_dangling_references_or_fix_hints_remain():
    for entry in load("patcheval_verified_cleaned.json"):
        text = entry.get("cve_description") or ""
        assert "insert a into" not in text, entry["cve_id"]
        assert "where is the" not in text, entry["cve_id"]
        assert not DANGLING.search(text), entry["cve_id"]
        assert not FIX_HINT.search(text), entry["cve_id"]


def test_answer_fields_are_metadata_not_prompt_input():
    """Answer metadata is expected; the prompt boundary is what must hold.

    If a future upstream refresh drops these fields, DATASET_NOTES.md must be
    updated. The prompt-side guarantee lives in test_patch_agent_runner.py.
    """
    entries = load("patcheval_verified.json")

    assert len(entries) == 230
    for entry in entries:
        assert entry.get("patch_url")
        assert entry.get("fix_func")
        for region in entry.get("vul_func") or []:
            assert region.get("file_path")
            assert region.get("commit")
