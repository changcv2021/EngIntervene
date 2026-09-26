"""Strict completeness checks, task/domain aggregates, and held-out Challenge."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from .io import read_rows, unique_rows, write_json


def summary(rows):
    if not rows:
        return {'count': 0, 'mean_score': None, 'strict_success_rate': None}
    return {'count': len(rows), 'mean_score': statistics.fmean(r['score']['score_fraction'] for r in rows),
            'strict_success_rate': statistics.fmean(r['score']['strict_success'] for r in rows),
            'judge_fallbacks': sum(r.get('judge_fallbacks', 0) for r in rows),
            'output_cap_hits': sum(r.get('output_cap_hit', False) for r in rows)}


def aggregate(items, scores, challenge):
    im, sm = unique_rows(items), unique_rows(scores)
    if set(im) != set(sm):
        raise ValueError('Missing, duplicate or unexpected score IDs')
    for iid, r in sm.items():
        score = r.get('score', {})
        value = score.get('score_fraction')
        if r.get('error') or not score.get('score_valid') or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid score: ' + iid)
        if (r['task_id'], r['domain']) != (im[iid]['task_id'], im[iid]['domain']):
            raise ValueError('Metadata mismatch')
    by_task = {t: summary([r for r in scores if r['task_id'] == t]) for t in ('T1', 'T2', 'T3', 'T4')}
    domains = sorted({r['domain'] for r in items})
    by_cell = {d: {t: summary([r for r in scores if r['domain'] == d and r['task_id'] == t])
                  for t in by_task} for d in domains}
    means = [x['mean_score'] for x in by_task.values() if x['count']]
    return {'all': summary(scores), 'by_task': by_task, 'domain_task': by_cell,
            'task_macro_average': statistics.fmean(means) if len(means) == 4 else None,
            'challenge': summary([r for r in scores if r['item_id'] in challenge]),
            'note': 'Null denotes N/A, never zero. T1 mean_score is accuracy; T2--T4 use weighted rubric scores.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--items', required=True)
    p.add_argument('--scores', required=True, nargs='+')
    p.add_argument('--challenge-ids', default='splits/challenge_test_ids.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    items = read_rows(args.items)
    scores = [r for path in args.scores for r in read_rows(path)]
    result = aggregate(items, scores, set(json.loads(Path(args.challenge_ids).read_text())))
    write_json(args.output, result)
    lines = ['# EngIntervene evaluation summary', '', '| Task | N | Accuracy / Atomic | Strict success |', '|---|---:|---:|---:|']
    fmt = lambda x: 'N/A' if x is None else f'{x:.2%}'
    for task, row in result['by_task'].items():
        lines.append(f'| {task} | {row["count"]} | {fmt(row["mean_score"])} | {fmt(row["strict_success_rate"])} |')
    lines += ['', f'Task macro-average: {fmt(result["task_macro_average"])}', '', result['note'],
              '', 'Judge scores are automatic, not independent human adjudication.']
    Path(args.output).with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
