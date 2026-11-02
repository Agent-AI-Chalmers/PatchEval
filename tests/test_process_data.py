import importlib.util
import json
from pathlib import Path

import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "patcheval/exp_agent/process_data.py"
    spec = importlib.util.spec_from_file_location("process_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_dataset(path, entries):
    path.write_text(
        json.dumps(
            [
                {"cve_id": cve, "programming_language": language, "image_url": image}
                for cve, language, image in entries
            ]
        )
    )
    return path


def write_patches(runner_output, names):
    patch_dir = runner_output / "patches"
    patch_dir.mkdir(parents=True)
    for name in names:
        (patch_dir / f"{name}.patch").write_text(f"diff for {name}\n")
    return patch_dir


def arguments(module, runner_output, dataset, output):
    return [
        module.__name__,
        "--runner-output",
        str(runner_output),
        "--process-data-path",
        str(output),
        "--test-data-path",
        str(dataset),
    ]


def test_converts_patches_with_dataset_metadata(tmp_path, monkeypatch):
    dataset = write_dataset(
        tmp_path / "dataset.json", [("CVE-1", "python", "image-1")]
    )
    runner_output = tmp_path / "run"
    write_patches(runner_output, ["CVE-1"])
    output = tmp_path / "eval.jsonl"
    module = load_module()
    monkeypatch.setattr(
        "sys.argv", arguments(module, runner_output, dataset, output)
    )

    assert module.main() == 0
    assert json.loads(output.read_text().splitlines()[0]) == {
        "cve": "CVE-1",
        "language": "python",
        "fix_patch": "diff for CVE-1\n",
        "image_url": "image-1",
    }


def test_rejects_patches_missing_from_dataset(tmp_path, monkeypatch):
    """Generation and evaluation datasets drifting apart must fail loudly.

    A silent subset would evaluate fewer cases than were generated, which looks
    like a lower success rate rather than a configuration error.
    """
    dataset = write_dataset(
        tmp_path / "dataset.json", [("CVE-1", "python", "image-1")]
    )
    runner_output = tmp_path / "run"
    write_patches(runner_output, ["CVE-1", "CVE-2"])
    output = tmp_path / "eval.jsonl"
    module = load_module()
    monkeypatch.setattr(
        "sys.argv", arguments(module, runner_output, dataset, output)
    )

    with pytest.raises(SystemExit) as failure:
        module.main()
    assert "CVE-2" in str(failure.value)
    assert str(dataset) in str(failure.value)
    assert not output.exists()


def test_rejects_duplicate_dataset_entries(tmp_path, monkeypatch):
    dataset = write_dataset(
        tmp_path / "dataset.json",
        [("CVE-1", "python", "image-1"), ("CVE-1", "python", "image-1")],
    )
    runner_output = tmp_path / "run"
    write_patches(runner_output, ["CVE-1"])
    output = tmp_path / "eval.jsonl"
    module = load_module()
    monkeypatch.setattr(
        "sys.argv", arguments(module, runner_output, dataset, output)
    )

    with pytest.raises(SystemExit) as failure:
        module.main()
    assert "duplicate" in str(failure.value)
