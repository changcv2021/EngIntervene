"""Normalize input-only JSONL or local Hugging Face Parquet; never load labels."""
import io
import json
from pathlib import Path
from .io import grouped_order, read_rows, unique_rows, write_rows

ALLOWED_INPUT_KEYS = ('question_benchmark', 'choices', 'response_contract', 'asset_ids')


def normalize(row):
    if 'input' in row:
        inp = {k: row['input'][k] for k in ALLOWED_INPUT_KEYS if k in row['input']}
        task = row['task_id']
    else:
        inp = {'question_benchmark': row['question'], 'choices': row.get('choices', []),
               'response_contract': row.get('response_contract', []), 'asset_ids': row['image_ids']}
        task = row['task']
    return {'item_id': row['item_id'], 'domain': row['domain'], 'task_id': task, 'input': inp}


def import_inputs(root, output, asset_root=None):
    """Only question-side fields and original input images are exported."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise RuntimeError('Input destination already exists; no overwrite')
    output.mkdir(parents=True)
    asset_root = Path(asset_root) if asset_root else root
    if (root / 'internal_items.jsonl').is_file():
        assets = {a['asset_id']: a for a in read_rows(root / 'assets.jsonl')}
        values = []
        for source in read_rows(root / 'internal_items.jsonl'):
            row = normalize(source)
            paths = []
            for aid in row['input']['asset_ids']:
                rel = assets[aid]['relative_path']
                path = next((p for p in [root / rel, asset_root / rel] if p.is_file()), None)
                if path is None:
                    raise FileNotFoundError(aid)
                paths.append(str(path.resolve()))
            row['images'] = paths
            values.append(row)
    else:
        # Arrow gives access to original encoded bytes: no JPEG recompression.
        import pyarrow.parquet as pq
        paths = sorted((root / 'data').glob('test-*.parquet'))
        if not paths:
            raise FileNotFoundError('Expected data/test-*.parquet or internal_items.jsonl')
        from PIL import Image
        values, seen = [], {}
        for path in paths:
            for source in pq.read_table(path).to_pylist():
                row = normalize(source)
                if len(source['images']) != len(row['input']['asset_ids']):
                    raise ValueError('Image ID/count mismatch')
                row['images'] = []
                for aid, image in zip(row['input']['asset_ids'], source['images']):
                    if not aid or Path(aid).name != aid:
                        raise ValueError('Unsafe image ID')
                    blob = image.get('bytes')
                    if blob is None:
                        raise ValueError('Embedded image bytes are required')
                    import hashlib
                    checksum = hashlib.sha256(blob).hexdigest()
                    if aid in seen and seen[aid][0] != checksum:
                        raise ValueError('Conflicting bytes for same image ID')
                    if aid not in seen:
                        with Image.open(io.BytesIO(blob)) as img:
                            ext = '.png' if img.format == 'PNG' else '.jpg' if img.format == 'JPEG' else '.img'
                        dest = output / 'images' / (aid + ext)
                        dest.parent.mkdir(exist_ok=True)
                        dest.write_bytes(blob)
                        seen[aid] = (checksum, str(dest.resolve()))
                    row['images'].append(seen[aid][1])
                values.append(row)
    unique_rows(values)
    write_rows(output / 'items.jsonl', values)
    return values


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--asset-root')
    args = p.parse_args()
    values = import_inputs(args.dataset, args.output, args.asset_root)
    membership = json.loads((Path(__file__).resolve().parents[1] / 'splits/membership.json').read_text())
    if {r['item_id'] for r in values} == set(membership):
        for split in ('train', 'validation', 'test'):
            selected = []
            for row in grouped_order(values, membership):
                m = membership[row['item_id']]
                if m['split'] == split:
                    selected.append({**row, 'split_group_id': m['group_id']})
            write_rows(Path(args.output) / f'{split}.jsonl', selected)
    print({'items': len(values)})


if __name__ == '__main__':
    main()
