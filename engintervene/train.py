"""Single-seed BF16 language-only LoRA with explicit assistant-label masking.

Benchmark files/labels are never passed to training. Smoke checkpoints cannot
be resumed into formal training. Formal resume restores optimizer and RNG.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import transformers
import peft
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForImageTextToText

from .io import digest, read_rows, write_json, load_config

HERE = Path(__file__).resolve().parent


def encode(processor, row, maximum, condition, image_budget):
    messages = [dict(row["messages"][0])]
    content = []
    for path in row["images"]:
        with Image.open(path) as original:
            img = original.convert("RGB")
        pixels = img.width * img.height
        budget = min(max(pixels, image_budget["min_pixels"]), image_budget["max_pixels"])
        if budget != pixels:
            scale = math.sqrt(budget / pixels)
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
        content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": row["messages"][1]["content"]})
    messages.append({"role": "user", "content": content})
    options = dict(tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    prefix = processor.apply_chat_template(messages, add_generation_prompt=True, **options)
    full = processor.apply_chat_template(messages + [row["messages"][-1]], add_generation_prompt=False, **options)
    n = prefix["input_ids"].shape[-1]
    if not torch.equal(full["input_ids"][:, :n], prefix["input_ids"]):
        raise RuntimeError(f"{row['sample_id']}: chat-template prefix mismatch; refusing unsafe label masking")
    if full["input_ids"].shape[-1] > maximum:
        raise RuntimeError(f"{row['sample_id']}: exceeds {maximum}; no silent truncation")
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    if condition == "format":
        answer = row["messages"][-1]["content"]
        encoded = processor.tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
        tokens = encoded["input_ids"]
        if full["input_ids"][0, n:n + len(tokens)].tolist() != tokens:
            raise RuntimeError(f"{row['sample_id']}: cannot align JSON structural labels exactly")
        if row["task_id"] == "T1":
            prefix_end, suffix_start = answer.index("[") + 1, answer.rindex("]")
        else:
            if not answer.startswith('{"answer":"') or not answer.endswith('"}'):
                raise RuntimeError("Unexpected canonical JSON wrapper")
            prefix_end, suffix_start = len('{"answer":"'), len(answer) - 2
        for i, (start, end) in enumerate(encoded["offset_mapping"]):
            # A token touching answer content is never supervised, even if it
            # also contains a JSON quote/punctuation (tokenizer boundary case).
            structural = end > start and (end <= prefix_end or start >= suffix_start)
            if not structural:
                labels[0, n + i] = -100
    if int((labels != -100).sum()) < 2:
        raise RuntimeError("No assistant supervision")
    full["labels"] = labels
    # These values would be cast to BF16 in move() anyway. Cast cached image
    # features early so two thousand image contexts do not exhaust CPU RAM.
    return {k: (v.to(dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in full.items()}


def move(row):
    return {k: (v.to(device="cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) if isinstance(v, torch.Tensor) else v) for k, v in row.items()}


def loss_value(model, row):
    batch = move(row)
    labels = batch.pop("labels")
    positions = torch.nonzero(labels[0, 1:] != -100, as_tuple=False).flatten()
    targets = labels[0, positions + 1]
    # Compute logits only at supervised prediction positions. This is exactly
    # shifted causal CE, without allocating a full image-context × vocabulary
    # logits tensor (particularly important for multi-image T4 questions).
    logits = model(**batch, logits_to_keep=positions, use_cache=False).logits
    return torch.nn.functional.cross_entropy(logits[0].float(), targets)


def evaluate(model, rows):
    model.eval()
    loss_sum, count = 0.0, 0
    with torch.no_grad():
        for row in rows:
            n = int((row["labels"][:, 1:] != -100).sum())
            loss = float(loss_value(model, row))
            if not math.isfinite(loss):
                raise RuntimeError("Non-finite development loss")
            loss_sum += loss * n
            count += n
    model.train()
    return loss_sum / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--mode", choices=["smoke", "train"], required=True)
    parser.add_argument("--condition", choices=["format", "industry"], required=True)
    parser.add_argument("--config", default="configs/training.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA device per training process")
    cfg = load_config(config_path)
    mc = cfg["models"][args.model_index]
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.set_float32_matmul_precision("high")
    root = Path(cfg["data_root"])
    data = root / "data/internal_grouped_v1/sft_train_validation_only"
    manifest = json.loads((data / "manifest.json").read_text())
    for split in ("train", "validation"):
        if digest(data / f"{split}.jsonl") != manifest["splits"][split]["sha256"]:
            raise RuntimeError("Training data changed after freeze")
    out = root / "runs" / mc["key"] / (f"{args.condition}_smoke" if args.mode == "smoke" else f"{args.condition}_seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    report = out / "progress.json"
    global ACTIVE_REPORT
    ACTIVE_REPORT = report
    if (out / "completed.json").exists():
        print("Already complete; refusing duplicate training", flush=True)
        return
    config_hash, script_hash = digest(config_path), digest(Path(__file__))
    identity = {"config_sha256": config_hash, "script_sha256": script_hash, "data_manifest_sha256": digest(data / "manifest.json"), "condition": args.condition}
    progress = {"model_key": mc["key"], "condition": args.condition, "stage": args.mode, "status": "loading", "job_id": os.environ.get("SLURM_JOB_ID", "local"), "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"), "seed": seed, "output": str(out), **identity}
    write_json(report, progress)
    processor = AutoProcessor.from_pretrained(mc["model_path"], local_files_only=True, use_fast=True)
    train_raw, dev_raw = read_rows(data / "train.jsonl"), read_rows(data / "validation.jsonl")
    if any(r.get("split") != "train" for r in train_raw) or any(r.get("split") != "validation" for r in dev_raw):
        raise RuntimeError("Unexpected split label in train/validation export")
    if {r["sample_id"] for r in train_raw}.intersection(r["sample_id"] for r in dev_raw):
        raise RuntimeError("Train/validation overlap")
    for row in train_raw + dev_raw:
        if row.get("split") not in ("train", "validation"):
            raise RuntimeError("Test or unknown data admitted to training")
        for path, sha in zip(row["images"], row["image_sha256"], strict=True):
            if digest(path) != sha:
                raise RuntimeError("Training image integrity mismatch")
    started = time.time()
    # All training/validation records are tokenization-audited even in smoke mode.
    train = [encode(processor, row, cfg["max_sequence_length"], args.condition, cfg["image_budget"]) for row in train_raw]
    dev = [encode(processor, row, cfg["max_sequence_length"], args.condition, cfg["image_budget"]) for row in dev_raw]
    lengths = [r["input_ids"].shape[-1] for r in train]
    write_json(out / "input_audit.json", {"rows": len(train), "max_sequence": max(lengths), "min_sequence": min(lengths), "supervised_tokens": sum(int((r["labels"] != -100).sum()) for r in train), "prompt_loss_masked": True, "silent_truncation": False, "condition": args.condition, "test_rows_loaded": 0})
    schedule = json.loads((data / "sampling_schedule.json").read_text())
    if digest(data / "sampling_schedule.json") != manifest["sampling_schedule_sha256"]:
        raise RuntimeError("Task-balanced sampling schedule changed")
    iid_to_index = {r["sample_id"]: i for i, r in enumerate(train_raw)}
    if args.mode == "smoke":
        # Stress-test the longest member of each task, plus max-image record.
        indices = {max([i for i, r in enumerate(train_raw) if r["task_id"] == t], key=lambda i: lengths[i]) for t in ("T1", "T2", "T3", "T4")}
        indices.add(max(range(len(train_raw)), key=lambda i: len(train_raw[i]["images"])))
        indices = sorted(indices)
        train = [train[i] for i in indices]
        dev_indices = [next(i for i, r in enumerate(dev_raw) if r["task_id"] == t) for t in ("T1", "T2", "T3", "T4")]
        dev = [dev[i] for i in dev_indices]
        del train_raw, dev_raw
    loader = AutoModelForMultimodalLM if mc["auto_class"] == "multimodal_lm" else AutoModelForImageTextToText
    base = loader.from_pretrained(mc["model_path"], local_files_only=True, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", low_cpu_mem_usage=True)
    base.config.use_cache = False
    targets = [name for name, module in base.named_modules() if isinstance(module, torch.nn.Linear) and ".language_model." in ("." + name + ".") and ".layers." in name and not any(s in name for s in ("vision", "visual", "projector", "lm_head"))]
    if not targets:
        raise RuntimeError("Cannot identify language-only LoRA modules")
    resume_path = out / "latest.json"
    resume = json.loads(resume_path.read_text()) if resume_path.exists() else None
    if resume and resume["identity"] != identity:
        raise RuntimeError("Resume identity differs; refusing mixed-protocol training")
    if resume:
        model = PeftModel.from_pretrained(base, resume["checkpoint"], is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], target_modules=targets, bias="none"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not params or any("lora_" not in n or any(t in n for t in ("vision", "visual", "projector")) for n, p in params):
        raise RuntimeError("Unexpected non-language trainable weights")
    provenance = {**mc, **identity, "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "seed": seed, "trainable_parameters": sum(p.numel() for _, p in params), "lora_target_modules": targets, "checkpoint_selection": cfg["formal_selection"]}
    model_index = Path(mc["model_path"]) / "model.safetensors.index.json"
    provenance["model_config_sha256"] = digest(Path(mc["model_path"]) / "config.json")
    if model_index.exists():
        provenance["model_weight_index_sha256"] = digest(model_index)
    if mc.get("manifest_path"):
        provenance["source_manifest"] = json.loads(Path(mc["manifest_path"]).read_text())
    write_json(out / "run_manifest.json", provenance)
    optimizer = torch.optim.AdamW([p for _, p in params], lr=cfg["learning_rate"], weight_decay=0.01)
    epochs = 1 if args.mode == "smoke" else cfg["epochs"]
    accumulation = 1 if args.mode == "smoke" else cfg["gradient_accumulation"]
    steps_per_epoch = math.ceil(len(train) / accumulation)
    total_steps = epochs * steps_per_epoch
    step, epoch_start, offset, best = 0, 0, 0, {"loss": float("inf")}
    if resume:
        state = torch.load(Path(resume["checkpoint"]) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        step, epoch_start, offset, best = state["step"], state["epoch"], state["offset"], state["best"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        random.setstate(state["python_rng"])

    def checkpoint(epoch, position):
        cp = out / f"checkpoint-{step:06d}"
        cp.mkdir(exist_ok=True)
        model.save_pretrained(cp)
        processor.save_pretrained(cp)
        torch.save({"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "offset": position, "best": best, "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}, cp / "training_state.pt")
        write_json(resume_path, {"checkpoint": str(cp), "identity": identity})
        return cp

    initial_loss = None
    if args.mode == "smoke":
        initial_loss = evaluate(model, dev)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_started = time.time()
    for epoch in range(epoch_start, epochs):
        if args.mode == "smoke":
            order = list(range(len(train)))
        else:
            order = [iid_to_index[iid] for iid in schedule["epochs"][epoch]]
        begin = offset if epoch == epoch_start else 0
        for first in range(begin, len(order), accumulation):
            group = order[first:first + accumulation]
            rate = min(1.0, (step + 1) / max(1, round(total_steps * 0.03)))
            cosine = 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps)))
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["learning_rate"] * rate * cosine
            losses = []
            for idx in group:
                loss = loss_value(model, train[idx])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {step}")
                (loss / len(group)).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0, error_if_nonfinite=True)
            if step == 0 and float(norm) == 0:
                raise RuntimeError("No training gradient reached LoRA")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            progress.update(status="training", step=step, total_steps=total_steps, epoch=epoch + 1, examples_seen=epoch * len(train) + first + len(group), loss=sum(losses) / len(losses), gradient_norm=float(norm), peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, elapsed_seconds=time.time() - started)
            write_json(report, progress)
            with (out / "training_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
            if args.mode == "train" and step % 50 == 0 and first + len(group) < len(order):
                checkpoint(epoch, first + len(group))
        dev_loss = evaluate(model, dev)
        cp = out / f"checkpoint-{step:06d}"
        if dev_loss < best["loss"]:
            best = {"loss": dev_loss, "checkpoint": str(cp), "epoch": epoch + 1}
        checkpoint(epoch + 1, 0)
        write_json(out / "best_checkpoint.json", best)
        offset = 0
    result = {**progress, "status": "smoke_passed" if args.mode == "smoke" else "training_complete", "best": best, "initial_dev_loss": initial_loss, "final_dev_loss": dev_loss, "training_seconds": time.time() - train_started, "seconds_per_microbatch": (time.time() - train_started) / (epochs * len(train)), "benchmark_evaluated": False}
    write_json(out / "completed.json", result)
    write_json(report, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        p = globals().get("ACTIVE_REPORT")
        if p is not None:
            previous = json.loads(p.read_text()) if p.exists() else {}
            previous.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:2000])
            write_json(p, previous)
        raise
