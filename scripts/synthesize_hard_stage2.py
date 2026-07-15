"""Compress long math reasoning with a teacher model and verify the result.

Uses an OpenAI-compatible Chat Completions endpoint.  DeepSeek V4 is the
default provider; DashScope remains selectable through command-line options.
Outputs are append-only and keyed by candidate_id, so interrupted runs can
resume safely.
"""

import argparse
import hashlib
import http.client
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer
import certifi
import requests

from sft_data_utils import format_math_sft


COMPRESS_SYSTEM = """你是数学监督微调数据的高级编辑器。你的任务不是自由发挥，而是把给定的冗长解答改写成正确、紧凑、可验证的推理。严格输出 JSON，不要输出 Markdown。"""

VERIFY_SYSTEM = """你是严格的数学训练数据审校员。检查推理是否逻辑正确、是否足以支持指定答案。不要因为答案字段与参考答案相同就放宽推理要求。严格输出 JSON，不要输出 Markdown。"""


class AuthenticationError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hard_stage2_candidates/train_candidates.jsonl")
    parser.add_argument("--output-dir", default="data/hard_stage2_synthesis_sample20")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-model", default="deepseek-v4-flash")
    parser.add_argument("--verifier-model", default="deepseek-v4-pro")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        help="OpenAI-compatible endpoint; e.g. https://api.deepseek.com",
    )
    parser.add_argument(
        "--api-key-env", default="DEEPSEEK_API_KEY",
        help="Environment-variable name holding the API key (never stored in outputs).",
    )
    parser.add_argument("--tokenizer", default=str(Path.home() / ".cache/modelscope/hub/models/Qwen/Qwen3-8B"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-reasoning-tokens", type=int, default=256)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-source-chars", type=int, default=12000)
    parser.add_argument("--teacher-max-tokens", type=int, default=1800)
    parser.add_argument("--verifier-max-tokens", type=int, default=800)
    return parser.parse_args()


def candidate_id(record):
    payload = f"{record.get('source', '')}\n{record.get('instruction', '')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def normalize_teacher_text(text):
    """Convert double-escaped paragraph markers without touching LaTeX commands."""
    text = text.replace(r"\r\n", "\n").replace(r"\n\n", "\n\n")
    # A remaining literal \n is treated as a newline only when it is not the
    # beginning of a lowercase LaTeX command such as \neq, \not, or \nabla.
    return re.sub(r"\\n(?=[^a-z])", "\n", text)


def compact_source_reasoning(text, max_chars):
    """Bound teacher context while retaining the setup and final derivation."""
    text = str(text)
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 5
    tail = max_chars - head
    return (
        text[:head]
        + "\n\n[中间重复/冗长部分已为节省上下文而省略；请根据题目、保留部分与参考答案重建严谨解法。]\n\n"
        + text[-tail:]
    )


def parse_model_json(content):
    """Parse a JSON object even when a provider adds a code fence/preamble."""
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError as original:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise original


def api_call(base_url, api_key, model, system, user, max_tokens=3000, thinking_type=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if thinking_type:
        payload["thinking"] = {"type": thinking_type}
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "sft-project-hard-data-builder/1.0",
        "Connection": "close",
    }
    try:
        response = requests.post(
            url, headers=headers, json=payload, timeout=(30, 300), verify=certifi.where()
        )
        if response.status_code in {401, 403}:
            raise AuthenticationError(
                f"API authentication failed ({response.status_code}); check the key in the configured "
                "--api-key-env and --base-url"
            )
        response.raise_for_status()
        raw = response.text
        if not raw.strip():
            raise requests.exceptions.ConnectionError("empty API response")
        payload = response.json()
    except requests.exceptions.HTTPError:
        raise
    content = payload["choices"][0]["message"]["content"]
    return parse_model_json(content), payload.get("usage", {})


def compression_prompt(record, feedback="", max_source_chars=12000):
    suffix = f"\n上一次审校反馈：{feedback}\n请据此修正。" if feedback else ""
    return f"""请压缩下面的数学样本。

要求：
1. 先判断原始推理是否能支持给定参考答案；如果题面或推理不可修复，status 返回 rejected。
2. 删除重复计算、自我怀疑、无效尝试、元话语和多次复算。
3. 保留得到答案所必需的公式、推导与关键论证，不能只给答案。
4. 最终答案必须严格保持为参考答案，不得自行修改。
5. reasoning 目标为 600-1500 个 Qwen token，复杂证明可略长，但完整样本必须能进入 2048 token。
6. reasoning 中不要出现 <think>、<answer> 或 ChatML 标签。
7. 严格返回：
{{"status":"accepted|rejected","reasoning":"...","answer":"...","reject_reason":"..."}}

题目：
{record['instruction']}

原始冗长推理（可能已确定性截取首尾以避免上下文浪费）：
{compact_source_reasoning(record['reasoning'], max_source_chars)}

已核验参考答案：
{record['answer']}
{suffix}"""


def verifier_prompt(record, reasoning):
    return f"""请审校下面的压缩数学推理。

检查：
1. 每个关键步骤是否正确；
2. 是否遗漏了导致结论无法成立的关键步骤；
3. 是否真正支持指定参考答案；
4. 是否存在无根据假设、题面缺失或循环论证；
5. 是否足够紧凑，且没有大量重复。

严格返回：
{{"valid":true|false,"supports_answer":true|false,"missing_key_step":true|false,"error":"..."}}

题目：
{record['instruction']}

压缩推理：
{reasoning}

指定参考答案：
{record['answer']}"""


def select_sample(records, limit, seed):
    rng = random.Random(seed)
    groups = {}
    for record in records:
        source = str(record.get("source", "unknown"))
        groups.setdefault(source, []).append(record)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    preferred = ["amc_aime", "EduChat-Math", "Haijian/Advanced-Math",
                 "meta-math/GSM8K_zh", "gavinluo/applied_math"]
    while len(selected) < limit and any(groups.values()):
        for source in preferred:
            if len(selected) >= limit:
                break
            if groups.get(source):
                selected.append(groups[source].pop())
        for source in sorted(groups):
            if len(selected) >= limit:
                break
            if source not in preferred and groups[source]:
                selected.append(groups[source].pop())
    return selected


def load_done(*paths):
    done = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)["candidate_id"])
    return done


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_existing_accepted(path, tokenizer):
    if not path.exists():
        return
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    changed = False
    for record in records:
        normalized = normalize_teacher_text(record["reasoning"])
        if normalized != record["reasoning"]:
            record["reasoning"] = normalized
            record["compressed_reasoning_tokens"] = len(tokenizer(
                normalized, add_special_tokens=False
            )["input_ids"])
            record["compressed_sft_tokens"] = len(tokenizer(
                format_math_sft(record), add_special_tokens=False
            )["input_ids"])
            changed = True
    if changed:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, path)


def migrate_transient_rejections(rejected_path, transient_path):
    if not rejected_path.exists():
        return
    permanent = []
    transient = []
    with open(rejected_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("reject_reason", "")).startswith("api_or_parse_error"):
                transient.append(record)
            else:
                permanent.append(record)
    if not transient:
        return
    with open(transient_path, "a", encoding="utf-8") as handle:
        for record in transient:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary = rejected_path.with_suffix(rejected_path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in permanent:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, rejected_path)


def main():
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"{args.api_key_env} is not set")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    records = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    selected = select_sample(records, args.limit, args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    transient_path = output_dir / "transient_errors.jsonl"
    normalize_existing_accepted(accepted_path, tokenizer)
    migrate_transient_rejections(rejected_path, transient_path)
    done = load_done(accepted_path, rejected_path)

    for position, record in enumerate(selected, 1):
        cid = candidate_id(record)
        if cid in done:
            continue
        feedback = ""
        usage = []
        final_reject = "max_attempts_exhausted"
        last_short_draft = None
        for attempt in range(1, args.max_attempts + 1):
            try:
                compressed, use = api_call(
                    args.base_url, api_key, args.teacher_model, COMPRESS_SYSTEM,
                    compression_prompt(record, feedback, args.max_source_chars),
                    max_tokens=args.teacher_max_tokens, thinking_type="disabled",
                )
                usage.append({"stage": "compress", "attempt": attempt, **use})
                if compressed.get("status") != "accepted":
                    final_reject = f"teacher_rejected:{compressed.get('reject_reason', '')}"
                    break
                reasoning = normalize_teacher_text(
                    str(compressed.get("reasoning", "")).strip()
                )
                # The teacher often canonicalizes an equivalent mathematical
                # answer (for example, changes LaTex spacing).  The dataset's
                # answer is already independently repaired, so keep it
                # verbatim and let the verifier judge whether the new
                # reasoning supports that fixed reference answer.
                reasoning_tokens = len(tokenizer(reasoning, add_special_tokens=False)["input_ids"])
                formatted = {**record, "reasoning": reasoning, "answer": record["answer"]}
                total_tokens = len(tokenizer(
                    format_math_sft(formatted), add_special_tokens=False
                )["input_ids"])
                if reasoning_tokens < args.min_reasoning_tokens:
                    last_short_draft = {
                        "teacher_reasoning": reasoning,
                        "teacher_reasoning_tokens": reasoning_tokens,
                        "teacher_formatted_sft_tokens": total_tokens,
                    }
                    feedback = f"推理过短，只有 {reasoning_tokens} token，缺少必要步骤。"
                    final_reject = "reasoning_too_short"
                    continue
                if total_tokens > args.max_length:
                    feedback = f"完整样本为 {total_tokens} token，必须压缩到 {args.max_length} 以内。"
                    final_reject = "formatted_too_long"
                    continue

                verified, use = api_call(
                    args.base_url, api_key, args.verifier_model, VERIFY_SYSTEM,
                    verifier_prompt(record, reasoning), max_tokens=args.verifier_max_tokens,
                    thinking_type="disabled",
                )
                usage.append({"stage": "verify", "attempt": attempt, **use})
                valid = (
                    verified.get("valid") is True
                    and verified.get("supports_answer") is True
                    and verified.get("missing_key_step") is False
                )
                if not valid:
                    feedback = str(verified.get("error", "verifier rejected"))
                    final_reject = f"verifier_rejected:{feedback}"
                    continue

                append_jsonl(accepted_path, {
                    "candidate_id": cid,
                    "instruction": record["instruction"],
                    "reasoning": reasoning,
                    "answer": record["answer"],
                    "source": record.get("source", "unknown"),
                    "original_sft_token_length": record.get("sft_token_length"),
                    "compressed_reasoning_tokens": reasoning_tokens,
                    "compressed_sft_tokens": total_tokens,
                    "teacher_model": args.teacher_model,
                    "verifier_model": args.verifier_model,
                    "verifier": verified,
                    "usage": usage,
                })
                print(f"[{position}/{len(selected)}] accepted {cid}: {total_tokens} tokens", flush=True)
                break
            except (
                urllib.error.URLError,
                http.client.RemoteDisconnected,
                http.client.IncompleteRead,
                ConnectionResetError,
                TimeoutError,
                ssl.SSLError,
                json.JSONDecodeError,
                requests.exceptions.RequestException,
                KeyError,
            ) as exc:
                final_reject = f"api_or_parse_error:{type(exc).__name__}:{exc}"
                feedback = final_reject
                time.sleep(min(2 ** attempt, 8))
        else:
            target_path = transient_path if final_reject.startswith("api_or_parse_error") else rejected_path
            append_jsonl(target_path, {
                "candidate_id": cid, "source": record.get("source", "unknown"),
                "instruction": record["instruction"], "reject_reason": final_reject,
                "usage": usage, **(last_short_draft or {}),
            })
            print(f"[{position}/{len(selected)}] rejected {cid}: {final_reject}", flush=True)
            continue
        if final_reject.startswith("teacher_rejected"):
            append_jsonl(rejected_path, {
                "candidate_id": cid, "source": record.get("source", "unknown"),
                "instruction": record["instruction"], "reject_reason": final_reject,
                "usage": usage,
            })
            print(f"[{position}/{len(selected)}] rejected {cid}: {final_reject}", flush=True)


if __name__ == "__main__":
    main()
