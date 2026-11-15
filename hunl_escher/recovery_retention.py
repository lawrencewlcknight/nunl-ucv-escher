"""Keep only the latest published recovery checkpoint in a run directory."""
import json
import re
import shutil
from pathlib import Path

from .state_storage import SHARDED_MANIFEST, SHARDED_STATE_TYPE, training_state_storage_info


_STATE_NAME = re.compile(r"state_([0-9]{6,})")


def prune_recovery_checkpoints(root: Path, keep: str) -> list[str]:
    """Run only after save verification and atomic publication of latest.json.

    Restrict deletion to older, recognized recovery directories in this run.
    Policy files, unrelated directories, temporary saves and symlinks are left
    alone. A failed cleanup raises, with the newly published recovery intact.
    """
    root = Path(root)
    current = _STATE_NAME.fullmatch(keep)
    if current is None or (root / keep).is_symlink():
        raise ValueError("Invalid retained recovery checkpoint")
    if json.loads((root / "latest.json").read_text()).get("state") != keep:
        raise ValueError("Publish the new recovery checkpoint before pruning")
    # The saver has just verified every checksum. Check the manifest and file
    # presence here without adding another multi-gigabyte checksum pass.
    training_state_storage_info(root / keep, verify=False)
    removed = []
    for candidate in sorted(root.iterdir()):
        match = _STATE_NAME.fullmatch(candidate.name)
        if (match is None or int(match[1]) >= int(current[1])
                or candidate.is_symlink() or not candidate.is_dir()):
            continue
        manifest_path = candidate / SHARDED_MANIFEST
        if manifest_path.is_symlink():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict) or manifest.get("type") != SHARDED_STATE_TYPE:
            continue
        shutil.rmtree(candidate)
        removed.append(candidate.name)
    return removed
