from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_temp_output(target_path: str, content: str, suffix: str = ".tmp") -> Path:
    target = Path(target_path)
    ensure_parent(target)
    temp_path = target.with_suffix(target.suffix + suffix)
    temp_path.write_text(content, encoding="utf-8")
    return temp_path


def promote_output_if_valid(temp_path: Path, live_path: str, is_valid: bool) -> bool:
    live = Path(live_path)
    ensure_parent(live)

    if not is_valid:
        if temp_path.exists():
            temp_path.unlink()
        return False

    shutil.move(str(temp_path), str(live))
    return True


def save_snapshot(live_path: str, snapshot_path: str) -> Optional[Path]:
    live = Path(live_path)
    snapshot = Path(snapshot_path)

    if not live.exists():
        return None

    ensure_parent(snapshot)
    shutil.copy2(str(live), str(snapshot))
    return snapshot


def restore_last_known_good(snapshot_path: str, live_path: str) -> bool:
    snapshot = Path(snapshot_path)
    live = Path(live_path)

    if not snapshot.exists():
        return False

    ensure_parent(live)
    shutil.copy2(str(snapshot), str(live))
    return True
