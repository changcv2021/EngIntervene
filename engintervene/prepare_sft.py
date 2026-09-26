"""Use the exact frozen source-grouped split, not a new item-random split."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .io import digest, grouped_order, read_rows, unique_rows, write_json, write_rows
from .protocol import prompt, target_answer


def prepare(items_path, labels_path, membership_path, schedule_path, output, system):
    items = read_rows(items_path)
    by_id = unique_rows(items)
    membership = json.loads(Path(membership_path).read_text())
    if set(by_id) != set(membership):
        raise ValueError('Items must match frozen membership exactly')
    items = grouped_order(items, membership)
    # Values themselves are private local runtime artifacts, never code-package contents.
    allowed = {i for i, m in membership.items() if m['split'] in ('train', 'validation')}
    labels = {}
    with Path(labels_path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['item_id'] in allowed:
                if row['item_id'] in labels:
                    raise ValueError('Duplicate label')
                labels[row['item_id']] = row
    if set(labels) != allowed:
        raise ValueError('Missing train/validation references')
    group_splits, image_splits = {}, {}
    for item in items:
        m = membership[item['item_id']]
        if m['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        old = group_splits.setdefault(m['group_id'], m['split'])
        if old != m['split']:
            raise ValueError('Source-group leakage')
        for image in item['images']:
            checksum = digest(image)
            if image_splits.setdefault(checksum, m['split']) != m['split']:
                raise ValueError('Image-byte leakage')
    output = Path(output)
    if output.exists():
        raise RuntimeError('Prepared SFT directory already exists')
    output.mkdir(parents=True)
    schedule = json.loads(Path(schedule_path).read_text())
    train_ids = {i for i in allowed if membership[i]['split'] == 'train'}
    for epoch in schedule['epochs']:
        if len(epoch) != len(train_ids) or not set(epoch) <= train_ids:
            raise ValueError('Invalid or leaking sampling schedule')
    manifests = {}
    for split in ('train', 'validation'):
        values = []
        for item in items:
            iid = item['item_id']
            if membership[iid]['split'] != split:
                continue
            values.append({'sample_id': iid, 'split': split,
                'split_group_id': membership[iid]['group_id'], 'task_id': item['task_id'], 'domain': item['domain'],
                'images': item['images'], 'image_sha256': [digest(p) for p in item['images']],
                'messages': [{'role': 'system', 'content': system},
                             {'role': 'user', 'content': prompt(item)},
                             {'role': 'assistant', 'content': target_answer(item, labels[iid])}]})
        write_rows(output / f'{split}.jsonl', values)
        manifests[split] = {'count': len(values), 'task_counts': dict(Counter(r['task_id'] for r in values)),
                            'sha256': digest(output / f'{split}.jsonl')}
    write_json(output / 'sampling_schedule.json', schedule)
    write_json(output / 'manifest.json', {'splits': manifests, 'test_rows_exported': 0,
        'membership_sha256': digest(membership_path), 'sampling_schedule_sha256': digest(output / 'sampling_schedule.json')})
    return manifests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--items', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--membership', default='splits/membership.json')
    p.add_argument('--schedule', default='splits/sampling_schedule.json')
    p.add_argument('--output', required=True)
    p.add_argument('--eval-config', default='configs/evaluation.json')
    args = p.parse_args()
    system = json.loads(Path(args.eval_config).read_text())['system_prompt']
    print(prepare(args.items, args.labels, args.membership, args.schedule, args.output, system))


if __name__ == '__main__':
    main()
