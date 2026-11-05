import argparse
import concurrent.futures as cf
import fcntl
import functools
import hashlib
import io
import json
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Registry endpoints are kept here so mirror changes do not touch download logic.
# Official metadata remains the trust anchor; mirrors only supply blob bytes.
OFFICIAL_REGISTRY_URL = "https://ghcr.io"
NJU_REGISTRY_URL = "https://ghcr.nju.edu.cn"
REPOSITORY = "patcheval-cve/patcheval-cve"
OFFICIAL_TOKEN_URL = (
    "https://ghcr.io/token?service=ghcr.io"
    "&scope=repository:patcheval-cve/patcheval-cve:pull"
)

# Keep manifests, logs, and resumable parts outside the repository checkout.
DEFAULT_STATE = Path.home() / ".local/state/patcheval/image-pull"
AUTH = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(s):
    print(time.strftime("%H:%M:%S", time.gmtime()), s, flush=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


CHUNK_SIZE = 1024 * 1024
RETRYABLE_ERRORS = (OSError, urllib.error.URLError, TimeoutError)
MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"


def file_matches_digest(path, size, digest):
    if not path.exists() or path.stat().st_size != size:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == digest.split(":")[1]


def resolve_blob_url(origin_url, token):
    """Resolve the official authenticated endpoint without forwarding its token."""
    metadata = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        origin_url, headers={"Authorization": "Bearer " + token}
    )
    try:
        with metadata.open(request, timeout=20):
            return origin_url
    except urllib.error.HTTPError as redirect:
        if redirect.code not in (302, 307):
            raise
        return redirect.headers["Location"]


def transfer_part(url, start, end, size, part_path):
    """Download one requested range; reject truncated or mismatched responses."""
    # Signed CDN URLs and mirror requests must not receive the GHCR token.
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with AUTH.open(request, timeout=30) as r, part_path.open("wb") as out:
        expected_range = f"bytes {start}-{end}/{size}"
        if r.status == 206:
            if r.headers.get("Content-Range") != expected_range:
                raise OSError("Unexpected range response")
        elif r.status != 200 or start != 0 or end != size - 1:
            raise OSError("Server ignored requested range")
        deadline = time.monotonic() + 240
        remaining = end - start + 1
        while remaining:
            b = r.read(min(1024 * 1024, remaining))
            if not b:
                raise OSError("Truncated response")
            out.write(b)
            remaining -= len(b)
            if time.monotonic() > deadline:
                raise TimeoutError("Part deadline")
    if part_path.stat().st_size != end - start + 1:
        raise OSError("Part size mismatch")


def download_part(start, *, digest, size, work, get_url):
    end = min(start + CHUNK_SIZE, size) - 1
    part_path = work / f'{digest.split(":")[1]}.{start}.part'
    # Resume across runs: keep complete ranges and download only missing ranges.
    # Incomplete ranges restart from their beginning; there is no byte-level resume.
    if part_path.exists() and part_path.stat().st_size == end - start + 1:
        log(f"RESUME {work.name} offset={start}: reusing complete part")
        return part_path
    temporary = part_path.with_suffix(".downloading")
    force_refresh = False
    for attempt in range(5):
        try:
            transfer_part(get_url(force_refresh), start, end, size, temporary)
            temporary.replace(part_path)
            log(
                f"PART {work.name} {start // CHUNK_SIZE + 1}/{(size + CHUNK_SIZE - 1) // CHUNK_SIZE}"
            )
            return part_path
        except RETRYABLE_ERRORS as exc:
            force_refresh = isinstance(exc, urllib.error.HTTPError) and exc.code == 403
            temporary.unlink(missing_ok=True)
            log(f"RETRY {work.name} offset={start} {type(exc).__name__}: {exc}")
            if attempt < 4:
                time.sleep(2)
    raise RuntimeError(f"Part failed: {work.name} offset={start}")


def assemble_verified_blob(parts, target, digest):
    """Verify the entire blob before trusting any parts reused from disk."""
    checksum = hashlib.sha256()
    with target.open("wb") as output:
        for part_path in parts:
            with part_path.open("rb") as source:
                while block := source.read(CHUNK_SIZE):
                    checksum.update(block)
                    output.write(block)
    if checksum.hexdigest() != digest.split(":")[1]:
        target.unlink()
        # Size alone cannot detect damaged cached parts. Discard them on hash failure.
        for part_path in parts:
            part_path.unlink(missing_ok=True)
        raise ValueError("Official blob hash mismatch; cached parts discarded")
    for part_path in parts:
        part_path.unlink()
    return target


def fetch_blob(digest, size, work, token, source):
    """Reuse a verified blob, resume its parts, then assemble and verify it."""
    target = work / digest.split(":")[1]
    if file_matches_digest(target, size, digest):
        return target
    origin_url = f"{OFFICIAL_REGISTRY_URL}/v2/{REPOSITORY}/blobs/{digest}"
    url = (
        resolve_blob_url(origin_url, token)
        if source == "official"
        else f"{NJU_REGISTRY_URL}/v2/{REPOSITORY}/blobs/{digest}"
    )
    refresh_at = time.monotonic() + 240
    url_lock = threading.Lock()

    def get_url(force_refresh):
        nonlocal url, token, refresh_at
        with url_lock:
            if source == "official" and (
                force_refresh or time.monotonic() > refresh_at
            ):
                with AUTH.open(OFFICIAL_TOKEN_URL, timeout=20) as response:
                    token = json.load(response)["token"]
                url = resolve_blob_url(origin_url, token)
                refresh_at = time.monotonic() + 240
            return url

    download = functools.partial(
        download_part, digest=digest, size=size, work=work, get_url=get_url
    )
    with cf.ThreadPoolExecutor(max_workers=32) as pool:
        parts = list(pool.map(download, range(0, size, CHUNK_SIZE)))
    return assemble_verified_blob(parts, target, digest)


def inspect_image(image):
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def load_verified_manifest(path, digest):
    """Use the saved official manifest as the trust anchor for every blob."""
    manifest = path.read_bytes()
    if "sha256:" + hashlib.sha256(manifest).hexdigest() != digest:
        raise ValueError("Manifest mismatch")
    return manifest, json.loads(manifest)


def get_official_token():
    with AUTH.open(OFFICIAL_TOKEN_URL, timeout=30) as response:
        return json.load(response)["token"]


def write_image_archive(archive, entry, manifest, manifest_data, blobs):
    # Preserve the original manifest bytes and image name when importing into Docker.
    with tarfile.open(archive, "w") as tar:

        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add("oci-layout", b'{"imageLayoutVersion":"1.0.0"}')
        add(
            "index.json",
            json.dumps(
                {
                    "schemaVersion": 2,
                    "manifests": [
                        {
                            "mediaType": manifest_data["mediaType"],
                            "digest": entry["digest"],
                            "size": len(manifest),
                            "annotations": {
                                "org.opencontainers.image.ref.name": entry["image"]
                            },
                        }
                    ],
                }
            ).encode(),
        )
        add("blobs/sha256/" + entry["digest"].split(":")[1], manifest)
        for digest, p in blobs.items():
            tar.add(p, arcname="blobs/sha256/" + digest.split(":")[1])


def import_verified_image(archive, entry):
    subprocess.run(["docker", "load", "-i", str(archive)], check=True)
    data = inspect_image(entry["image"])
    if data is None or data.get("Descriptor", {}).get("digest") != entry["digest"]:
        raise ValueError("Loaded manifest mismatch")
    return data


def record_success(path, entry, data, source):
    save_json(path, dict(entry, status="success", source=source, image_id=data["Id"]))


def pull_locked_image(entry, state_dir, source):
    image = entry["image"]
    tag = image.split(":")[-1]
    manifest, manifest_data = load_verified_manifest(
        state_dir / (tag + ".manifest.json"), entry["digest"]
    )
    existing = inspect_image(image)
    if existing and existing.get("Descriptor", {}).get("digest") == entry["digest"]:
        record_success(
            state_dir / (tag + ".result.json"), entry, existing, "local verified"
        )
        log("VERIFIED EXISTING " + image)
        return
    log("START " + image)
    # Stable filenames here are the checkpoints reused by download_part().
    work = state_dir / "direct-segments" / tag
    work.mkdir(parents=True, exist_ok=True)
    token = get_official_token() if source == "official" else None
    blobs = {
        blob["digest"]: fetch_blob(blob["digest"], blob["size"], work, token, source)
        for blob in [manifest_data["config"], *manifest_data["layers"]]
    }
    archive = work / "image.oci.tar"
    write_image_archive(archive, entry, manifest, manifest_data, blobs)
    data = import_verified_image(archive, entry)
    record_success(state_dir / (tag + ".result.json"), entry, data, source)
    log("SUCCESS " + image)
    # Keep checkpoints on failure; remove them only after verified import.
    for path in work.iterdir():
        path.unlink()
    work.rmdir()


def pull(entry, *, state_dir, source):
    tag = entry["image"].split(":")[-1]
    # Hold the lock through import and release it even if a download fails.
    with (state_dir / (tag + ".direct.lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pull_locked_image(entry, state_dir, source)


def pull_with_retries(entry, *, state_dir, source):
    for attempt in range(10):
        try:
            pull(entry, state_dir=state_dir, source=source)
            return True
        except Exception as exc:
            log(f'IMAGE RETRY {entry["image"]}: {type(exc).__name__}: {exc}')
            if attempt < 9:
                time.sleep(10)
    return False


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def prepare_official_manifests(images_file, state_dir):
    """Pin every tag against GHCR before downloading bytes from any mirror.

    Reuse saved pins across restarts so a moved upstream tag cannot change a
    partially completed run. Only this PatchEval repository and single manifests
    are supported; reject other inputs rather than silently downloading a subset.
    """
    images = list(
        dict.fromkeys(
            line.strip()
            for line in images_file.read_text().splitlines()
            if line.strip()
        )
    )
    if not images:
        raise ValueError("Image list is empty")
    prefix = "ghcr.io/" + REPOSITORY + ":"
    for image in images:
        if (
            not image.startswith(prefix)
            or not image.removeprefix(prefix)
            or any(
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                for c in image.removeprefix(prefix)
            )
        ):
            raise ValueError(f"Unsupported image reference: {image}")
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "manifests.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pins_path = state_dir / "official-digests.json"
        cached = (
            {entry["image"]: entry for entry in json.loads(pins_path.read_text())}
            if pins_path.exists()
            else {}
        )
        entries = []
        token = None
        for image in images:
            tag = image.removeprefix(prefix)
            manifest_path = state_dir / (tag + ".manifest.json")
            entry = cached.get(image)
            if entry is not None:
                _, data = load_verified_manifest(manifest_path, entry["digest"])
            else:
                if token is None:
                    token = get_official_token()
                request = urllib.request.Request(
                    f"{OFFICIAL_REGISTRY_URL}/v2/{REPOSITORY}/manifests/{tag}",
                    headers={
                        "Authorization": "Bearer " + token,
                        "Accept": MANIFEST_MEDIA_TYPE,
                    },
                )
                with AUTH.open(request, timeout=30) as response:
                    manifest = response.read()
                    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
                    if response.headers.get("Docker-Content-Digest") != digest:
                        raise ValueError(f"Official manifest digest mismatch: {image}")
                data = json.loads(manifest)
                entry = {"image": image, "digest": digest}
                temporary = manifest_path.with_suffix(".tmp")
                temporary.write_bytes(manifest)
                temporary.replace(manifest_path)
            if (
                data.get("mediaType") != MANIFEST_MEDIA_TYPE
                or "config" not in data
                or "layers" not in data
            ):
                raise ValueError(f"Unsupported manifest format: {image}")
            entries.append(entry)
            cached[image] = entry
            save_json(pins_path, list(cached.values()))
        return entries


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pull PatchEval images using official manifest digests and verified segmented downloads. State stays outside the checkout by default."
    )
    parser.add_argument(
        "--images-file", type=Path, default=Path(__file__).with_name("images.txt")
    )
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source", choices=("official", "nju"), default="official")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def main():
    args = parse_args()
    state_dir = args.state_dir.expanduser().resolve()
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    entries = prepare_official_manifests(args.images_file, state_dir)
    download = functools.partial(
        pull_with_retries, state_dir=state_dir, source=args.source
    )
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, entries))
    log(f"COMPLETE success={sum(results)} failed={len(results)-sum(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
