# Dataset notes

## What this file is

`patcheval_verified.json` is a task dataset with evaluation metadata. Fields such as
`vul_func` (file, line range, snippet, fix commit), `fix_func` and `patch_url` contain the
answer by design. That is expected and harmless here: the patch runner never sends them to
an agent — the prompt is built from `cve_id`, `cve_description` and `image_url` only — and
`test_prompt_does_not_leak_answer_fields` guards that boundary.

The text of `cve_description` is a different matter: it is the whole task statement, so
damage there changes the task.

## Prose defects

All of these already exist in `patcheval_verified.json`; none were introduced by the
cleaning pass.

| Kind | Count | CVEs | Present in |
| --- | ---: | --- | --- |
| Lost token in a sentence | 2 | CVE-2021-21384 "insert a into the payload"; CVE-2021-43798 "where is the plugin ID" | original and cleaned |
| Dangling advisory reference | 3 | CVE-2021-21384, CVE-2021-32804, CVE-2020-26294 | original and cleaned |
| Raw markdown link left behind | 2 | CVE-2024-29041, CVE-2021-32701 | original and cleaned |
| Fenced code block collapsed to one line | 2 | CVE-2022-35949, CVE-2024-0243 | original and cleaned |
| Typo | 1 | CVE-2021-43798 "iss vulnerable" | original and cleaned |
| Changelog residue | 1 | "No further changes are required." | original and cleaned |

Eight distinct descriptions are affected. The lost-token row is the serious one:
CVE-2021-21384 is about the missing byte, so as written the task cannot be answered from
the description alone. The MITRE record carries the same broken sentence; only the advisory
has the original wording.

## What the cleaning pass did

`patcheval_verified_cleaned.json` differs from the original in 169 descriptions:

- removed 83 fix hints ("before version 1.1.3", "patched in ...", "upgrade to ...") — now 0;
- removed 5 double spaces left by sentence deletion;
- no artifact was introduced: the two files have the same counts for leftover links,
  collapsed fences, empty parentheses and spaces before punctuation.

## In-place changes applied to the cleaned file

Before: `sha256 1fed05ba4adb5e34f42879e5f835eb8063e2e15c2ddc3150fcc3a2aca740badf`
After: `sha256 aae5d0b8042cc4f6d78de3f56539f7409a0e95a892163921f49140c3d46ac1b6`

| CVE | Before | After | Source |
| --- | --- | --- | --- |
| CVE-2021-21384 | `insert a into the payload. For an example see the referenced GitHub Security Advisory.` | `insert a null character into the payload.` | [GHSA-f2rp-38vg-j3gh](https://github.com/ericcornelissen/shescape/security/advisories/GHSA-f2rp-38vg-j3gh) |
| CVE-2021-43798 | `<grafana_host_url>/public/plugins//`, where is the plugin ID | `<grafana_host_url>/public/plugins/<plugin-id>/`, where `<plugin-id>` is the plugin ID | [GHSA-8pjx-jj86-j47p](https://github.com/grafana/grafana/security/advisories/GHSA-8pjx-jj86-j47p) |
| CVE-2021-32804 | ...with absolute paths. See referenced GitHub Advisory for details. CVE-2021-32803... | ...with absolute paths. CVE-2021-32803... | [CVE record](https://cveawg.mitre.org/api/cve/CVE-2021-32804) |
| CVE-2020-26294 | ...configuration information, see referenced GHSA for an example. It is recommended... | ...configuration information. It is recommended... | [CVE record](https://cveawg.mitre.org/api/cve/CVE-2020-26294) |

Left alone on purpose: CVE-2021-37712 names `GHSA-qq89-hq3f-393p`, so it is not dangling;
the collapsed fences keep their tokens, only the layout is lost; the two raw links still
carry the same information.

Regenerate the edits (idempotent, fails loudly if the source text moved):

```bash
cd patcheval/datasets
python apply_description_fixes.py
```

## Method and limitations

Defects were found with targeted patterns over `cve_description`: "insert an? into",
"where is the", `](http` fragments, collapsed fences, and version wording. This is
heuristic, not exhaustive: a systematic check would fetch every upstream advisory and diff
it, which was not done.

## Comparability

The cleaned file was edited in place, so its hash no longer matches the published PatchEval
input. Results must not be compared against published leaderboard numbers. Every agent and
configuration in this project sees the same edited text, so internal comparisons stay fair.
