from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_existing_ids(path: Path, id_field: str = "id") -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    try:
        for row in iter_jsonl(path):
            ex_id = row.get(id_field)
            if isinstance(ex_id, str):
                ids.add(ex_id)
    except Exception:
        # If file is partially written / corrupted, do not crash the whole run.
        pass
    return ids
