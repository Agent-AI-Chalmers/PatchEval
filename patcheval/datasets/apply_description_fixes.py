"""Restore description text that markdown stripping destroyed in the verified dataset.

This script edits patcheval_verified_cleaned.json in place. It is idempotent: a
patch that has already been applied is reported as "already applied" instead of
failing, and a patch whose source text is missing raises immediately.

Every replacement cites the advisory it was taken from. Version numbers and
"fixed in" wording are never reintroduced: the cleaned dataset intentionally
keeps no fix hints.
"""

import hashlib
import json
from pathlib import Path

TARGET = Path(__file__).with_name("patcheval_verified_cleaned.json")

TICK = chr(96)


def code(value: str) -> str:
    return TICK + value + TICK


PATCHES = [
    {
        "cve": "CVE-2021-21384",
        "old": "insert a into the payload. For an example see the referenced GitHub Security Advisory.",
        "new": "insert a null character into the payload.",
        "reason": (
            "The advisory text links 'null character'; markdown stripping removed "
            "both the link and the words. The dangling advisory sentence is removed "
            "because its referent is not part of the input."
        ),
        "source": "https://github.com/ericcornelissen/shescape/security/advisories/GHSA-f2rp-38vg-j3gh",
    },
    {
        "cve": "CVE-2021-43798",
        "old": code("<grafana_host_url>/public/plugins//") + ", where is the plugin ID for any installed plugin.",
        "new": code("<grafana_host_url>/public/plugins/<plugin-id>/")
        + ", where "
        + code("<plugin-id>")
        + " is the plugin ID for any installed plugin.",
        "reason": (
            "The upstream record lost '<plugin-id>' in two places; the advisory "
            "keeps both the path segment and the variable name."
        ),
        "source": "https://github.com/grafana/grafana/security/advisories/GHSA-8pjx-jj86-j47p",
    },
    {
        "cve": "CVE-2021-32804",
        "old": "with absolute paths. See referenced GitHub Advisory for details. CVE-2021-32803",
        "new": "with absolute paths. CVE-2021-32803",
        "reason": "Dangling advisory reference: the link was stripped from the input.",
        "source": "https://cveawg.mitre.org/api/cve/CVE-2021-32804",
    },
    {
        "cve": "CVE-2020-26294",
        "old": "retrieve configuration information, see referenced GHSA for an example. It is recommended",
        "new": "retrieve configuration information. It is recommended",
        "reason": "Dangling advisory reference: the link was stripped from the input.",
        "source": "https://cveawg.mitre.org/api/cve/CVE-2020-26294",
    },
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    raw = TARGET.read_bytes()
    entries = json.loads(raw.decode("utf-8"))
    by_cve = {entry["cve_id"]: entry for entry in entries}

    applied = skipped = 0
    for patch in PATCHES:
        entry = by_cve[patch["cve"]]
        text = entry.get("cve_description") or ""
        if patch["new"] in text:
            skipped += 1
            continue
        if text.count(patch["old"]) != 1:
            raise SystemExit(
                f"{patch['cve']}: source fragment not found exactly once; "
                f"the dataset changed underneath this patch (source: {patch['source']})"
            )
        entry["cve_description"] = text.replace(patch["old"], patch["new"])
        applied += 1

    if applied:
        TARGET.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"before: sha256 {digest(raw)}")
    print(f"after : sha256 {digest(TARGET.read_bytes())}")
    print(f"applied={applied} already_applied={skipped} total={len(PATCHES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
