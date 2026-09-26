"""Historical judge decoding and retry algorithm."""
import json
import torch
from .protocol import valid_judgment

def move_to_device(inputs: dict, device: torch.device) -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def generate_text(model, processor, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "严格执行给定 rubric；输出必须是单行 JSON。"}],
        },
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = move_to_device(inputs, model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    generated = output[:, prompt_tokens:]
    return processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def judge_criterion(
    model, processor, prompt: str, criterion_id: str, response: str
) -> tuple[str, int, bool]:
    attempts = [
        (prompt, 160),
        (
            prompt
            + "\n上一次输出未通过格式或证据校验。重新检查：只输出一行 JSON；"
            + "若 met=true，evidence 必须是回答中 2–80 字符的连续逐字片段；否则判 false 并留空。",
            128,
        ),
        (
            prompt
            + "\n此前输出再次未通过硬校验。不得拼接不连续文本，不得输出超过80字符的证据。"
            + "如果无法从 candidate_answer 逐字复制一个短片段完整支持 criterion，必须返回 met=false 和空 evidence。",
            128,
        ),
    ]
    for attempt, (attempt_prompt, max_tokens) in enumerate(attempts):
        raw = generate_text(model, processor, attempt_prompt, max_tokens)
        if valid_judgment(raw, criterion_id, response):
            return raw, attempt, False
    # Fail closed: an invalid positive judgment must never award credit or abort
    # a multi-thousand-item run. The fallback is valid, conservative, and logged.
    fallback = json.dumps(
        {"criterion_id": criterion_id, "met": False, "evidence": ""},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return fallback, len(attempts) - 1, True
