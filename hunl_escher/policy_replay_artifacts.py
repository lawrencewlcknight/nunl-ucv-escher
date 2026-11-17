"""Read only average-policy replay from a trusted sharded recovery checkpoint."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

from .checkpointing import sha256_file
from .state_storage import SHARDED_STATE_TYPE, STATE_TYPE
from .storage_transfer import run_transfer

FIELDS = {"infostate": "infostate_buf", "q_value": "q_value_buf",
          "q_value_mask": "q_value_mask_buf", "iteration": "iteration_buf"}


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Checkpoint path escapes its directory")
    return path


def verify_file(path, record):
    if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")


def read_metadata(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if (manifest.get("type") != SHARDED_STATE_TYPE or manifest.get("schema_version") != 1
            or manifest.get("complete") is not True):
        raise ValueError("Incomplete or unsupported recovery manifest")
    path = safe_path(root, manifest["metadata_path"])
    verify_file(path, dict(size_bytes=manifest["metadata_size_bytes"], sha256=manifest["metadata_sha256"]))
    # Like the existing recovery loader, only use trusted locally produced artifacts.
    metadata = torch.load(path, map_location="cpu", weights_only=False)
    if metadata.get("type") != STATE_TYPE or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported training-state metadata")
    return manifest, metadata


def policy_records(manifest, metadata):
    records = {r["path"]: r for r in manifest["array_shards"]}
    state = metadata["average_policy_trainer"]["buffer"]
    result = {}
    for name in FIELDS:
        marker = state[name]
        if not isinstance(marker, dict) or set(marker) != {"__hunl_numpy_shard__"}:
            raise ValueError("Policy replay must use numpy shards")
        result[name] = records[marker["__hunl_numpy_shard__"]]
    return result


def load_policy_replay(root):
    """Verify selected shards and memory-map them; no critic buffers or solver."""
    manifest, metadata = read_metadata(root)
    state = metadata["average_policy_trainer"]["buffer"]
    records = policy_records(manifest, metadata)
    arrays = {}
    for name, record in records.items():
        path = safe_path(root, record["path"])
        verify_file(path, record)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError("Policy replay shape or dtype mismatch")
        if array.dtype != np.float32 or len(array) != state["size"]:
            raise ValueError("Invalid policy replay rows")
        arrays[FIELDS[name]] = array
    n = state["size"]
    if n < 1 or state["cur_id"] < n:
        raise ValueError("Invalid policy reservoir size")
    actions = len(metadata["game"]["action_menu"])
    if (arrays["q_value_buf"].shape != (n, actions) or arrays["q_value_mask_buf"].shape != (n, actions)
            or arrays["iteration_buf"].shape != (n, 1) or arrays["infostate_buf"].ndim != 2):
        raise ValueError("Invalid policy replay columns")
    buffer = SimpleNamespace(**arrays, cur_id=state["cur_id"], buffer_size=n, action_size=actions)
    return buffer, metadata, manifest


def prepare(source, output, *, metadata_only=False):
    """Fetch only metadata and four policy shards, never an entire run directory."""
    if not source.startswith("gs://"):
        raise ValueError("Preparation requires a gs:// checkpoint directory")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def fetch(relative, record=None):
        path = safe_path(output, relative)
        if path.exists() and record is not None:
            try:
                verify_file(path, record)
                return
            except ValueError:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".download")
        uri = source.rstrip("/") + "/" + relative
        try:
            run_transfer(["gcloud", "storage", "cp", uri, str(temp)], timeout=3600)
        except subprocess.CalledProcessError as exc:
            if "HashMismatchError" not in (exc.stderr or ""):
                raise
            # Some local gcloud sliced downloads fail their CRC check. Stream
            # the immutable object instead, then still require the manifest SHA.
            if record is None:
                raise
            with temp.open("wb") as handle:
                subprocess.run(["gcloud", "storage", "cat", uri], stdout=handle,
                               check=True, timeout=3600)
        if record is not None:
            verify_file(temp, record)
        temp.replace(path)

    fetch("manifest.json")
    manifest = json.loads((output / "manifest.json").read_text())
    fetch(manifest["metadata_path"], dict(size_bytes=manifest["metadata_size_bytes"], sha256=manifest["metadata_sha256"]))
    manifest, metadata = read_metadata(output)
    records = policy_records(manifest, metadata)
    plan = dict(source=source, outer_iteration=metadata["solver"]["num_iteration"],
        rows=metadata["average_policy_trainer"]["buffer"]["size"],
        policy_replay_bytes=sum(r["size_bytes"] for r in records.values()),
        metadata_bytes=manifest["metadata_size_bytes"],
        files=[r["path"] for r in records.values()], metadata_only=metadata_only,
        partial_checkpoint=True, note="Policy replay subset; not a resumable full training state")
    (output / "policy_subset.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2), flush=True)
    if not metadata_only:
        for record in records.values():
            fetch(record["path"], record)
    return plan
