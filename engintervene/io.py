"""Small protocol utilities; no model or network access on import."""
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(path)


def load_config(path):
    def expand(x):
        if isinstance(x, dict):
            return {k: expand(v) for k, v in x.items()}
        if isinstance(x, list):
            return [expand(v) for v in x]
        if isinstance(x, str):
            value = os.path.expandvars(os.path.expanduser(x))
            if '${' in value:
                raise ValueError('Unresolved environment variable in configuration')
            return value
        return x
    return expand(json.loads(Path(path).read_text(encoding='utf-8')))


def unique_rows(rows, key='item_id'):
    ids = [r[key] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs')
    return {r[key]: r for r in rows}


def shard(rows, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid shard configuration')
    return rows[index::count]


def grouped_order(rows, membership):
    """Historical export: groups by smallest item ID, then IDs within each group."""
    first = {}
    for iid, member in membership.items():
        gid = member['group_id']
        first[gid] = min(first.get(gid, iid), iid)
    return sorted(rows, key=lambda row: (first[membership[row['item_id']]['group_id']], row['item_id']))


def start_output(path, identity):
    """Bind resumable output to exact input/config/model identities."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = path.with_suffix(path.suffix + '.identity.json')
    if path.exists() and not meta.exists():
        raise RuntimeError('Existing output has no provenance; choose a new output')
    if meta.exists():
        if json.loads(meta.read_text()) != identity:
            raise RuntimeError('Resume protocol mismatch; choose a new output')
    else:
        write_json(meta, identity)
    return read_rows(path) if path.exists() else []
