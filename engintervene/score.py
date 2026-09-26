"""Deterministic T1 + separate frozen Qwen3.5-4B rubric judge for T2--T4."""
import argparse
import json
from pathlib import Path
from .io import digest, read_rows, unique_rows, shard, start_output
from .protocol import t1_score, rubric_score, judge_prompt, PROTOCOL


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--items', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--rubrics', required=True)
    p.add_argument('--generations', required=True, nargs='+')
    p.add_argument('--judge-model', help='Local, unadapted Qwen3.5-4B directory; required for T2--T4')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    items = read_rows(args.items)
    item_map = unique_rows(items)
    generations = unique_rows([r for path in args.generations for r in read_rows(path)])
    if set(item_map) != set(generations):
        raise ValueError('Candidate outputs must cover exactly the requested item IDs')
    if any(r.get('error') for r in generations.values()):
        raise ValueError('Inference errors are not valid model responses')
    # These are scoring-side files; candidate inference never loads either file.
    labels, rubrics = unique_rows(read_rows(args.labels)), unique_rows(read_rows(args.rubrics))
    if not set(item_map) <= set(labels) or not set(item_map) <= set(rubrics):
        raise ValueError('Missing gold coverage')
    identity = {'items_sha256': digest(args.items), 'labels_sha256': digest(args.labels),
                'rubrics_sha256': digest(args.rubrics), 'generations_sha256': [digest(p) for p in args.generations],
                'judge_protocol': PROTOCOL, 'shard_index': args.shard_index, 'num_shards': args.num_shards,
                'judge_config_sha256': digest(Path(args.judge_model) / 'config.json') if args.judge_model else None}
    values = shard(items, args.shard_index, args.num_shards)
    existing = start_output(args.output, identity)
    if [r['item_id'] for r in existing] != [r['item_id'] for r in values[:len(existing)]]:
        raise ValueError('Invalid score prefix')
    if any(not r['score']['score_valid'] for r in existing):
        raise ValueError('Prior scoring errors require inspection')
    model = processor = None
    if any(r['task_id'] != 'T1' for r in values[len(existing):]):
        if not args.judge_model:
            raise ValueError('Specify the unadapted judge model')
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        from .judge_runtime import judge_criterion
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose one CUDA device for judging')
        processor = AutoProcessor.from_pretrained(args.judge_model, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(args.judge_model, local_files_only=True,
                    dtype=torch.bfloat16, device_map='cuda', attn_implementation='sdpa')
        model.eval()
        torch.manual_seed(0)
    with Path(args.output).open('a', encoding='utf-8') as stream:
        for item in values[len(existing):]:
            iid = item['item_id']
            candidate = generations[iid]
            response = str(candidate.get('response') or '')
            judgments, retries, fallbacks = [], 0, 0
            if item['task_id'] == 'T1':
                score = t1_score(item, labels[iid], response)
            else:
                for criterion in rubrics[iid]['criteria']:
                    raw, retried, fallback = judge_criterion(model, processor, judge_prompt(item, response, criterion),
                                                           str(criterion['criterion_id']), response)
                    retries += retried
                    fallbacks += int(fallback)
                    judgments.append({'criterion_id': criterion['criterion_id'], 'raw': raw})
                score = rubric_score(response, rubrics[iid], judgments)
            if not score['score_valid']:
                raise RuntimeError('Invalid score, not a model failure: ' + iid)
            stream.write(json.dumps({**candidate, 'item_id': iid, 'task_id': item['task_id'], 'domain': item['domain'],
                         'score': score, 'judge_retries': retries, 'judge_fallbacks': fallbacks,
                         'scoring_identity': identity}, ensure_ascii=False) + '\n')
            stream.flush()


if __name__ == '__main__':
    main()
