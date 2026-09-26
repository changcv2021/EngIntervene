"""Single-GPU local inference; supports independent shards and image ablations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from .io import digest, load_config, read_rows, unique_rows, shard, start_output
from .protocol import prompt, emitted_answer


def visual_mapping(items, mode, seed=42):
    if mode == 'text-only':
        return {r['item_id']: [] for r in items}
    if mode == 'full':
        return {r['item_id']: r['images'] for r in items}
    hashes = {p: digest(p) for r in items for p in r['images']}
    result = {}
    for row in items:
        own = {hashes[p] for p in row['images']}
        donors = [r for r in items if r['item_id'] != row['item_id']
                  and (r['domain'], r['task_id']) == (row['domain'], row['task_id'])
                  and not own.intersection(hashes[p] for p in r['images'])]
        if not donors:
            raise ValueError('No disjoint same-domain/same-task shuffled donor: ' + row['item_id'])
        donor = random.Random(f'{seed}:{row["item_id"]}').choice(sorted(donors, key=lambda r: r['item_id']))
        result[row['item_id']] = donor['images']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--items', required=True)
    p.add_argument('--model-key', required=True)
    p.add_argument('--train-config', default='configs/training.json')
    p.add_argument('--eval-config', default='configs/evaluation.json')
    p.add_argument('--condition', choices=['base', 'format', 'industry'], default='base')
    p.add_argument('--visual-mode', choices=['full', 'text-only', 'shuffled-image'], default='full')
    p.add_argument('--adapter-run', help='Completed SFT run directory; best checkpoint selected by validation only')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    import torch
    import transformers
    import peft
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForImageTextToText
    from peft import PeftModel
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one CUDA device per independent process')
    train_cfg, cfg = load_config(args.train_config), load_config(args.eval_config)
    mc = next(r for r in train_cfg['models'] if r['key'] == args.model_key)
    all_items = read_rows(args.items)
    unique_rows(all_items)
    image_map = visual_mapping(all_items, args.visual_mode)
    values = shard(all_items, args.shard_index, args.num_shards)
    adapter, adapter_hash = None, None
    if args.condition != 'base':
        run = Path(args.adapter_run) if args.adapter_run else Path(train_cfg['data_root']) / 'runs' / mc['key'] / f'{args.condition}_seed{train_cfg["seed"]}'
        done = json.loads((run / 'completed.json').read_text())
        best = json.loads((run / 'best_checkpoint.json').read_text())
        if done['status'] != 'training_complete' or done['best'] != best or done['condition'] != args.condition or done['model_key'] != mc['key']:
            raise ValueError('Invalid or mismatched validation-selected adapter')
        adapter = Path(best['checkpoint'])
        adapter_hash = digest(adapter / 'adapter_model.safetensors')
    model_path = Path(mc['model_path'])
    identity = {'items_sha256': digest(args.items), 'evaluation_config_sha256': digest(args.eval_config),
                'training_config_sha256': digest(args.train_config), 'model_key': mc['key'],
                'base_config_sha256': digest(model_path / 'config.json'), 'adapter_sha256': adapter_hash,
                'condition': args.condition, 'visual_mode': args.visual_mode,
                'shard_index': args.shard_index, 'num_shards': args.num_shards}
    prior = start_output(args.output, identity)
    expected = [r['item_id'] for r in values]
    if [r['item_id'] for r in prior] != expected[:len(prior)] or any(r.get('error') for r in prior):
        raise ValueError('Invalid existing output prefix')
    for row in all_items:
        if not row['images']:
            raise ValueError('Original multimodal input has no images')
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=True)
    loader = AutoModelForMultimodalLM if mc['auto_class'] == 'multimodal_lm' else AutoModelForImageTextToText
    torch.set_float32_matmul_precision('high')
    base = loader.from_pretrained(model_path, local_files_only=True, dtype=torch.bfloat16,
                                  device_map='cuda', attn_implementation='sdpa', low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False) if adapter else base
    model.eval().requires_grad_(False)
    with Path(args.output).open('a', encoding='utf-8') as stream:
        for row in values[len(prior):]:
            start = time.perf_counter()
            content = []
            for path in image_map[row['item_id']]:
                with Image.open(path) as original:
                    img = original.convert('RGB')
                pixels = img.width * img.height
                budget = min(max(pixels, cfg['image_budget']['min_pixels']), cfg['image_budget']['max_pixels'])
                if budget != pixels:
                    scale = math.sqrt(budget / pixels)
                    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
                content.append({'type': 'image', 'image': img})
            content.append({'type': 'text', 'text': prompt(row)})
            seed = int(hashlib.sha256(f'{cfg["seed_protocol_id"]}:{mc["key"]}:{row["item_id"]}:0'.encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            messages = [{'role': 'system', 'content': cfg['system_prompt']}, {'role': 'user', 'content': content}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors='pt', enable_thinking=False)
            n = inputs['input_ids'].shape[-1]
            cap = cfg['max_new_tokens'][row['task_id']]
            context = getattr(base.config.get_text_config(), 'max_position_embeddings', None)
            if not 0 < cap <= 4096 or n > cfg['max_prompt_tokens'] or (context and n + cap > context):
                raise ValueError('Context/output budget exceeded; no silent input truncation')
            inputs = {k: v.to(device='cuda', dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                      if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=cap, use_cache=True, **cfg['generation'])[:, n:]
            tokens = int(gen.shape[-1])
            if tokens > cap:
                raise RuntimeError('Backend exceeded hard output cap')
            clean = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            raw = processor.batch_decode(gen, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0].strip()
            response, parse = emitted_answer(clean, raw)
            record = {**identity, 'item_id': row['item_id'], 'task_id': row['task_id'], 'domain': row['domain'],
                'response': response, 'raw_response': raw, 'clean_response': clean, 'parse_status': parse,
                'prompt_tokens': int(n), 'generated_tokens': tokens, 'max_new_tokens': cap,
                'output_cap_hit': tokens >= cap, 'candidate_attempts': 1, 'seed': seed, 'error': None,
                'inference_seconds': time.perf_counter() - start,
                'input_image_sha256': [digest(p) for p in image_map[row['item_id']]],
                'software': {'torch': torch.__version__, 'transformers': transformers.__version__, 'peft': peft.__version__}}
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps({'item_id': row['item_id'], 'generated_tokens': tokens}), flush=True)


if __name__ == '__main__':
    main()
