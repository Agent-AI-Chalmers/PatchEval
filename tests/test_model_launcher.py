import importlib.util
import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "agent,prefix", [("traecli", "TRAE"), ("codex", "CODEX"), ("opencode", "OPENCODE")]
)
@pytest.mark.parametrize("explicit_batch", [False, True])
@pytest.mark.parametrize("prompt_key", [False, True])
@pytest.mark.parametrize("key_padding", ["", " \t"])
def test_model_uses_temporary_private_config(
    monkeypatch, agent, prefix, explicit_batch, prompt_key, key_padding
):
    path = Path(__file__).resolve().parents[1] / "patcheval/exp_agent/run_model.py"
    spec = importlib.util.spec_from_file_location("model_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/true")
    if prompt_key:
        monkeypatch.delenv("MODEL_API_KEY", raising=False)
        monkeypatch.setattr(
            module.getpass,
            "getpass",
            lambda prompt: key_padding + "synthetic-secret" + key_padding,
        )
    else:
        monkeypatch.setenv(
            "MODEL_API_KEY", key_padding + "synthetic-secret" + key_padding
        )
    monkeypatch.setattr(
        "sys.argv",
        [
            str(path),
            "--agent",
            agent,
            "--model",
            "test-model",
            "--base-url",
            "https://example.org/v1",
            "--reasoning-effort",
            "low",
            "--model-provider",
            "deepseek",
        ]
        + (
            ["--limit", "3", "--concurrency", "2", "--cve", "CVE-2018-7753"]
            if explicit_batch
            else []
        ),
    )
    configs = []

    def run(command, *, env, check):
        config = Path(env[prefix + "_CONFIG"])
        configs.append(config)
        assert os.stat(config).st_mode & 0o777 == 0o600
        if agent == "opencode":
            data = json.loads(config.read_text())
            assert data["model"] == "model-api/test-model"
            assert data["small_model"] == "model-api/test-model"
            assert data["provider"]["model-api"]["npm"] == "@ai-sdk/openai"
            assert data["provider"]["model-api"]["options"] == {
                "baseURL": "https://example.org/v1",
                "apiKey": "synthetic-secret",
            }
            assert data["provider"]["model-api"]["models"]["test-model"]["options"] == {
                "reasoningEffort": "low"
            }
        else:
            data = tomllib.loads(config.read_text())
            assert data["model"] == "test-model"
            assert "disable_response_storage" not in data
            if agent == "codex":
                assert data["model_reasoning_effort"] == "low"
            else:
                assert "model_reasoning_effort" not in data
            assert data["model_providers"]["model-api"]["wire_api"] == "responses"
            assert (
                data["model_providers"]["model-api"]["experimental_bearer_token"]
                == "synthetic-secret"
            )
        assert env["MODEL_API_MODEL"] == "test-model"
        assert env["MODEL_API_PROVIDER"] == "deepseek"
        assert env["LIMIT"] == ("3" if explicit_batch else "-1")
        assert env["CONCURRENCY"] == ("2" if explicit_batch else "4")
        assert env["CVE"] == ("CVE-2018-7753" if explicit_batch else "")
        assert "synthetic-secret" not in str(command)
        assert command[-2] == agent
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main() == 1
    assert not configs[0].exists()


def test_documented_quickstart_reaches_generation(monkeypatch):
    """The copyable command must satisfy the launcher's required arguments."""
    import shlex

    directory = Path(__file__).resolve().parents[1] / "patcheval/exp_agent"
    readme = (directory / "README.md").read_text()
    quickstart = readme.split("## Model API quick start", 1)[1]
    example = quickstart.split("```bash\n", 1)[1].split("```", 1)[0]
    example = example.replace("\\\n", " ")
    command = shlex.split(example)
    spec = importlib.util.spec_from_file_location(
        "quickstart_launcher", directory / "run_model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", command[1:])
    monkeypatch.setattr(module.shutil, "which", lambda name: "/bin/true")
    monkeypatch.setenv("MODEL_API_KEY", "synthetic-secret")
    calls = []

    def run(argv, *, env, check):
        calls.append(argv)
        assert env["MODEL_API_MODEL"] == "MODEL"
        assert env["MODEL_API_PROVIDER"] == "VENDOR"
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main() == 0
    assert len(calls) == 1
