import hashlib
import io
import json
from unittest.mock import Mock

import pytest

from scripts import pull_images_verified as downloader


def test_resume_skips_completed_range_and_redownloads_partial(tmp_path, monkeypatch):
    digest = "sha256:" + hashlib.sha256(b"data").hexdigest()
    part = tmp_path / (digest.split(":")[1] + ".0.part")
    part.write_bytes(b"data")
    transfer = Mock(
        side_effect=lambda url, start, end, size, path: path.write_bytes(b"data")
    )
    monkeypatch.setattr(downloader, "transfer_part", transfer)
    kwargs = dict(digest=digest, size=4, work=tmp_path, get_url=lambda _: "url")
    assert downloader.download_part(0, **kwargs) == part
    transfer.assert_not_called()
    part.write_bytes(b"da")
    assert downloader.download_part(0, **kwargs).read_bytes() == b"data"
    assert transfer.call_args.args[-1].suffix == ".downloading"


def test_small_blob_rejects_incorrect_content_range(tmp_path, monkeypatch):
    response = io.BytesIO(b"data")
    response.status = 206
    response.headers = {"Content-Range": "bytes 1-4/5"}
    monkeypatch.setattr(downloader.AUTH, "open", lambda *args, **kwargs: response)
    with pytest.raises(OSError, match="range"):
        downloader.transfer_part("https://example.invalid", 0, 3, 4, tmp_path / "part")


def test_bad_cached_parts_are_discarded(tmp_path):
    part = tmp_path / "part"
    part.write_bytes(b"bad")
    target = tmp_path / "blob"
    digest = "sha256:" + hashlib.sha256(b"good").hexdigest()
    with pytest.raises(ValueError, match="hash mismatch"):
        downloader.assemble_verified_blob([part], target, digest)
    assert not target.exists()
    assert not part.exists()


def test_fresh_state_pins_official_manifest_and_reuses_pin(tmp_path, monkeypatch):
    image = "ghcr.io/patcheval-cve/patcheval-cve:test"
    images = tmp_path / "images.txt"
    images.write_text(image + "\n")
    manifest = json.dumps(
        {"mediaType": downloader.MANIFEST_MEDIA_TYPE, "config": {}, "layers": []}
    ).encode()
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    response = io.BytesIO(manifest)
    response.headers = {"Docker-Content-Digest": digest}
    monkeypatch.setattr(downloader, "get_official_token", lambda: "token")
    open_mock = Mock(return_value=response)
    monkeypatch.setattr(downloader.AUTH, "open", open_mock)
    state = tmp_path / "state"
    assert downloader.prepare_official_manifests(images, state) == [
        {"image": image, "digest": digest}
    ]
    assert downloader.prepare_official_manifests(images, state) == [
        {"image": image, "digest": digest}
    ]
    assert open_mock.call_count == 1
    (state / "test.manifest.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Manifest mismatch"):
        downloader.prepare_official_manifests(images, state)


def test_official_digest_mismatch_does_not_create_pin(tmp_path, monkeypatch):
    images = tmp_path / "images.txt"
    images.write_text("ghcr.io/patcheval-cve/patcheval-cve:test\n")
    response = io.BytesIO(b"{}")
    response.headers = {"Docker-Content-Digest": "sha256:incorrect"}
    monkeypatch.setattr(downloader, "get_official_token", lambda: "token")
    monkeypatch.setattr(downloader.AUTH, "open", lambda *args, **kwargs: response)
    state = tmp_path / "state"
    with pytest.raises(ValueError, match="Official manifest digest mismatch"):
        downloader.prepare_official_manifests(images, state)
    assert not (state / "official-digests.json").exists()


def test_failed_transfer_never_becomes_resume_checkpoint(tmp_path, monkeypatch):
    digest = "sha256:" + hashlib.sha256(b"data").hexdigest()

    def interrupted(url, start, end, size, path):
        # Even a full-size write is untrusted if the transfer did not finish validation.
        path.write_bytes(b"data")
        raise OSError("connection reset")

    monkeypatch.setattr(downloader, "transfer_part", interrupted)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Part failed"):
        downloader.download_part(
            0, digest=digest, size=4, work=tmp_path, get_url=lambda _: "url"
        )
    assert list(tmp_path.iterdir()) == []
