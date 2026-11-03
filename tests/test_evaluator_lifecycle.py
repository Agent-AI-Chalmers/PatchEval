"""Exercise evaluator ownership without starting Docker or changing scoring."""

import importlib.util
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def evaluator(monkeypatch):
    path = (
        Path(__file__).resolve().parents[1] / "patcheval/evaluation/run_evaluation.py"
    )
    spec = importlib.util.spec_from_file_location("verified_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    client = Mock()
    monkeypatch.setattr(module.docker, "from_env", lambda: client)
    return module, client


def test_evaluation_labels_containers(evaluator):
    module, client = evaluator
    manager = module.DockerManager(
        logging.getLogger("test"), "CVE-test", "example.owner=owner"
    )
    assert manager.start_container("image", "container", "") == "container"
    assert client.containers.run.call_args.kwargs["labels"] == {
        "example.owner": "owner"
    }


@pytest.mark.parametrize("failure", ["start", "validation", "cleanup"])
def test_evaluation_reclaims_container_and_exposes_infrastructure_failure(
    evaluator, failure
):
    module, client = evaluator
    evaluation = module.Evaluation(
        logging.getLogger("test"), container_label="example.owner=owner"
    )
    manager = evaluation.docker_manager
    manager.start_container = Mock(
        return_value="container" if failure != "start" else None
    )
    evaluation._run_sh_cmd = Mock(return_value=(True, "ok"))
    if failure == "validation":
        evaluation._run_sh_cmd.side_effect = RuntimeError("Docker API failed")
    if failure == "cleanup":
        client.containers.get.return_value.remove.side_effect = RuntimeError(
            "cannot remove"
        )
    if failure == "start":
        result = evaluation.run_evaluation("CVE-test", "", "Python", "run", "image")
        assert result[0] is None and result[2] is None
    else:
        with pytest.raises(RuntimeError):
            evaluation.run_evaluation("CVE-test", "", "Python", "run", "image")
    client.containers.get.return_value.remove.assert_called_with(force=True)


@pytest.mark.parametrize("label", ["owner", "=owner"])
def test_evaluation_rejects_invalid_label(evaluator, label):
    module, _ = evaluator
    with pytest.raises(ValueError, match="KEY=VALUE"):
        module.DockerManager(logging.getLogger("test"), "CVE-test", label)


def test_evaluation_container_has_no_network(evaluator):
    module, client = evaluator
    manager = module.DockerManager(Mock(), "CVE-test")
    assert manager.start_container("image", "container", "patch") == "container"
    assert client.containers.run.call_args.kwargs["network_mode"] == "none"
