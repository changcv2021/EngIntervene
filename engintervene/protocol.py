"""Historical prompts and scoring algorithms, copied without semantic changes."""
import json
import re
import unicodedata
PROTOCOL = "qwen35_4b_atomic_criterion_verbatim_evidence_v1.1"

def extract_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def evidence_is_verbatim(response: str, evidence: object) -> bool:
    normalized_evidence = normalized_text(evidence)
    return len(normalized_evidence) >= 2 and normalized_evidence in normalized_text(response)


def rubric_score(response: str, rubric: dict, judgments: list[dict]) -> dict:
    criteria = rubric.get("criteria", [])
    by_id = {str(row.get("criterion_id")): row for row in criteria}
    decisions: dict[str, bool] = {}
    evidence: dict[str, str] = {}
    parsed_count = 0
    raw_outputs = []
    for judgment in judgments:
        raw = judgment["raw"]
        parsed = extract_json(raw)
        raw_outputs.append(raw)
        criterion_id = judgment["criterion_id"]
        if not parsed or str(parsed.get("criterion_id")) != criterion_id:
            continue
        parsed_count += 1
        quoted = str(parsed.get("evidence") or "")
        supported = parsed.get("met") is True and evidence_is_verbatim(response, quoted)
        decisions[criterion_id] = supported
        evidence[criterion_id] = quoted if supported else ""

    required = {
        criterion_id: float(row.get("weight", 1) or 1)
        for criterion_id, row in by_id.items()
        if row.get("type") != "critical_error" and float(row.get("weight", 1) or 1) > 0
    }
    critical = {}
    for criterion_id, row in by_id.items():
        weight = float(row.get("weight", 0) or 0)
        if row.get("type") == "critical_error" or weight < 0:
            critical[criterion_id] = abs(weight) if weight else abs(float(row.get("penalty", 0) or 0))
    total = sum(required.values())
    earned = sum(weight for criterion_id, weight in required.items() if decisions.get(criterion_id) is True)
    penalty = sum(weight for criterion_id, weight in critical.items() if decisions.get(criterion_id) is True)
    cap = float(rubric.get("score_cap", rubric.get("maximum_score", total)) or total)
    score_points = max(0.0, min(cap, earned - penalty)) if cap > 0 else None
    score = score_points / cap if score_points is not None else None
    strict_required = [str(value) for value in rubric.get("strict_success_required") or required]
    strict_forbidden = [str(value) for value in rubric.get("strict_success_forbidden") or critical]
    strict_success = (
        bool(strict_required)
        and all(decisions.get(criterion_id) is True for criterion_id in strict_required)
        and not any(decisions.get(criterion_id) is True for criterion_id in strict_forbidden)
    )
    group_results = {}
    for group in rubric.get("criterion_groups") or []:
        group_id = str(group.get("group_id"))
        members = [str(value) for value in group.get("members") or []]
        member_weights = [required.get(member, 0.0) for member in members]
        denominator = sum(member_weights)
        group_results[group_id] = {
            "members": members,
            "partial_fraction": (
                sum(weight for member, weight in zip(members, member_weights) if decisions.get(member) is True) / denominator
                if denominator > 0 else None
            ),
            "strict_all": bool(members) and all(decisions.get(member) is True for member in members),
        }
    return {
        "scoring_method": PROTOCOL,
        "score_valid": parsed_count == len(by_id) and total > 0 and 0 < cap <= total,
        "score_fraction": score,
        "required_points_earned": earned,
        "required_points_total": total,
        "score_cap": cap,
        "score_points_after_penalty": score_points,
        "critical_error_penalty": penalty,
        "criteria_met": decisions,
        "criteria_evidence": evidence,
        "criteria_count": len(by_id),
        "criteria_judged": parsed_count,
        "critical_error": any(decisions.get(criterion_id) is True for criterion_id in critical),
        "strict_success": strict_success,
        "strict_success_required": strict_required,
        "strict_success_forbidden": strict_forbidden,
        "criterion_group_results": group_results,
        "judge_parse_error": parsed_count != len(by_id),
        "judge_raw": raw_outputs,
    }


def judge_prompt(item: dict, response: str, criterion: dict) -> str:
    expected = {
        "criterion_id": str(criterion.get("criterion_id")),
        "met": False,
        "evidence": "",
    }
    payload = {
        "task": item["task_id"],
        "question": item["input"].get("question_benchmark"),
        "candidate_answer": response,
        "criterion": {
            "criterion_id": criterion.get("criterion_id"),
            "description": criterion.get("description"),
            "type": criterion.get("type"),
        },
    }
    return (
        "你是严谨、保守的工程 benchmark 原子评分器，本次只判断一个 criterion。"
        "只能依据 candidate_answer 判断，不得把 question 或 criterion 本身当作回答证据。"
        "对 required：候选回答中的证据必须语义蕴含 criterion 的完整工程命题；"
        "只出现相同名词、相关症状、泛泛提到需要检查、或只满足必要条件的一部分，一律 met=false。"
        "若 criterion 仍含多个必须同时成立的子命题，必须全部被候选回答明确满足。"
        "对 critical_error：仅当候选回答明确主张、批准或建议该错误时 met=true；"
        "没有提及正确做法不等于触发错误。"
        "met=true 时 evidence 必须逐字复制 candidate_answer 中最短的连续支持片段，"
        "去除空白后长度须为 2–80 字符；met=false 时 evidence 必须为空。"
        "禁止解释、Markdown 和额外字段，只返回一行 JSON："
        + json.dumps(expected, ensure_ascii=False)
        + "\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def valid_judgment(raw: str, criterion_id: str, response: str) -> bool:
    parsed = extract_json(raw)
    if not parsed or str(parsed.get("criterion_id")) != criterion_id:
        return False
    if not isinstance(parsed.get("met"), bool):
        return False
    evidence = str(parsed.get("evidence") or "")
    if parsed["met"]:
        compact_length = len("".join(evidence.split()))
        return 2 <= compact_length <= 80 and evidence_is_verbatim(response, evidence)
    return not evidence.strip()

def canonical_choice(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "").strip()).upper()
    text = re.sub(r"^(?:OPT(?:ION)?[-_: ]*)", "", text)
    return "".join(character for character in text if character.isalnum())


def parse_choices(response: str, valid_raw: list[str]) -> list[str]:
    valid = [canonical_choice(value) for value in valid_raw]
    valid_set = set(valid)
    parsed = extract_json(response)
    raw_values: list[object] = []
    if parsed:
        value = parsed.get("choice_ids", parsed.get("choice_id"))
        raw_values = value if isinstance(value, list) else ([] if value is None else [value])
    found = {canonical_choice(value) for value in raw_values} & valid_set
    found.discard("")
    if not found:
        normalized_response = unicodedata.normalize("NFKC", response)
        for raw, normalized in zip(valid_raw, valid):
            token = unicodedata.normalize("NFKC", str(raw).strip())
            if len(token) == 1 and token.isascii() and token.isalnum():
                match = re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", response, re.I)
            else:
                match = re.search(re.escape(token), normalized_response, re.I)
            if match:
                found.add(normalized)
    return [value for value in valid if value in found]


def t1_score(item: dict, label: dict, response: str) -> dict:
    valid_raw = [str(row.get("choice_id") or "") for row in item["input"].get("choices", [])]
    valid = [canonical_choice(value) for value in valid_raw]
    predicted = parse_choices(response, valid_raw)
    gold_raw = label.get("gold_structure", {}).get("correct_choice_ids")
    if not gold_raw:
        single = label.get("gold_structure", {}).get("correct_choice_id")
        gold_raw = [single] if single is not None else []
    gold_set = {canonical_choice(value) for value in gold_raw} - {""}
    gold = [value for value in valid if value in gold_set]
    score_valid = bool(gold) and len(valid) == len(set(valid)) and all(valid)
    fraction = 1.0 if score_valid and predicted == gold else 0.0
    return {
        "scoring_method": "deterministic_unicode_exact_choice_set_v2",
        "predicted_choice_ids": predicted,
        "gold_choice_ids": gold,
        "score_fraction": fraction,
        "score_valid": score_valid,
        "strict_success": fraction == 1.0,
    }

def prompt(item):
    value = item["input"]
    lines = [f"任务：{item['task_id']}", "问题：" + value["question_benchmark"]]
    if value.get("choices"):
        lines += ["选项："] + [f"{c['choice_id']}: {c['text']}" for c in value["choices"]]
    lines += ["回答要求："] + ["- " + s for s in value.get("response_contract", [])]
    if item["task_id"] == "T1":
        lines.append('最终只返回 JSON，例如 {"choice_ids":["A"]}；choice_ids 必须使用题面选项 ID。')
    else:
        lines.append('最终只返回 JSON：{"answer":"你的完整回答"}。回答应包含必要工程判断和理由。')
    return "\n".join(lines)


def target_answer(item, label):
    if item["task_id"] != "T1":
        answer = label.get("benchmark_reference_answer") or label.get("reference_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(f"Missing reference answer: {item['item_id']}")
        return json.dumps({"answer": answer.strip()}, ensure_ascii=False, separators=(",", ":"))
    gold = label.get("gold_structure", {})
    choices = gold.get("correct_choice_ids") or gold.get("correct_choice_id") or gold.get("correct_option")
    if choices is None:
        choices = label.get("benchmark_reference_answer")
    valid = {c["choice_id"] for c in item["input"].get("choices", [])}
    if isinstance(choices, str):
        if choices in valid:
            choices = [choices]
        else:
            choices = re.findall(r"\b[A-Z]\b", choices)
    if not isinstance(choices, list) or not choices or not set(choices).issubset(valid):
        raise RuntimeError(f"Unusable MCQ gold target: {item['item_id']}")
    return json.dumps({"choice_ids": sorted(set(choices))}, ensure_ascii=False, separators=(",", ":"))

def emitted_answer(clean, raw):
    if "</think>" in clean:
        final = clean.rsplit("</think>", 1)[1].strip()
        if final:
            return final, "final_after_think"
    # Preserve usable capped output; never invent an answer or auto-zero it.
    fallback = clean.strip() or re.sub(r"<\|[^>]+\|>", "", raw).strip()
    return fallback, "raw_emitted" if fallback else "empty_emitted"
