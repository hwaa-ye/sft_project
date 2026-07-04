"""
PRM Reward 模块：加载 PRM 模型，对推理链进行步骤级评分
替代 grpo_reward.py 中的规则 reward，用于 PRM+GRPO 实验
"""
import re, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class PRMRewardScorer:
    """加载 PRM 模型，对推理链的每个步骤打分，返回聚合 reward"""

    def __init__(self, prm_lora_path, base_model_path, score_head_path, device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, trust_remote_code=True, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载 base + LoRA
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map=None,
            trust_remote_code=True, local_files_only=True,
        ).to(device)
        base.config.output_hidden_states = True
        base = PeftModel.from_pretrained(base, prm_lora_path)
        base = base.merge_and_unload()  # 推理时 merge 加速

        hidden_size = base.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        ).to(device)
        self.score_head.load_state_dict(torch.load(score_head_path, map_location=device))
        self.score_head.eval()

        self.base = base
        self.model = base  # 保持引用

    def extract_steps(self, response):
        """从响应中提取推理步骤"""
        think_m = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        answer_m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        think = think_m.group(1).strip() if think_m else ""
        answer = answer_m.group(1).strip() if answer_m else ""

        if not think:
            return [], answer

        # 按双换行分步
        raw = think.strip()
        steps = [s.strip() for s in re.split(r'\n\s*\n', raw) if s.strip()]
        if len(steps) <= 1:
            steps = [s.strip() for s in raw.split('\n') if s.strip()]
        return steps, answer

    def score_step(self, question, step_text):
        """对单个步骤打分"""
        prompt = (
            f"<|im_start|>user\n"
            f"评估以下数学推理步骤的质量。\n\n"
            f"问题: {question}\n"
            f"步骤: {step_text}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                max_length=1024).to(self.device)

        with torch.no_grad():
            outputs = self.base(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1]  # [1, seq, hidden]
            seq_len = inputs["attention_mask"].sum().item() - 1
            last_token = last_hidden[0, seq_len]  # [hidden]
            score = self.score_head(last_token.unsqueeze(0)).squeeze().item()

        return float(score)

    def compute_reward(self, response, answer, question=""):
        """
        计算 PRM-based reward，返回与 grpo_reward.compute_reward 兼容的字典
        """
        steps, pred_answer = self.extract_steps(response)

        # 基础结构：检查是否有 think/answer 标签
        has_think = "</think>" in response
        has_answer = "</answer>" in response
        is_complete = has_think and has_answer

        if not steps:
            return {
                "total": 0.0,
                "accuracy": 0.0,
                "completeness": 0.0,
                "efficiency": 0.0,
                "prm_score": 0.0,
            }

        # 步骤级评分
        step_scores = [self.score_step(question, s) for s in steps]
        prm_score = sum(step_scores) / len(step_scores) if step_scores else 0.0

        # 答案是否正确（如果提供了 gold answer）
        accuracy = 0.0
        if answer:
            from grpo_reward import check_answer
            try:
                accuracy = 1.0 if check_answer(pred_answer or response, str(answer)) else 0.0
            except Exception:
                pass

        # 效率：基于步骤数的分段评分
        n_steps = len(steps)
        if n_steps <= 3:
            efficiency = 0.3  # 太短 → 可能跳步
        elif n_steps <= 8:
            efficiency = 1.0 - 0.5 * (n_steps - 3) / 5  # 3~8步线性递减
        else:
            efficiency = 0.5  # 太长

        completeness = 1.0 if is_complete else 0.0

        # 总 reward = PRM 评分主导 (0.7) + accuracy (0.2) + efficiency (0.1)
        total = 0.7 * prm_score + 0.2 * accuracy + 0.1 * efficiency

        return {
            "total": total,
            "accuracy": accuracy,
            "completeness": completeness,
            "efficiency": efficiency,
            "prm_score": prm_score,
            "step_scores": step_scores,
            "n_steps": n_steps,
        }


# ─── 测试入口 ───
if __name__ == "__main__":
    import sys, json

    scorer = PRMRewardScorer(
        prm_lora_path="output/prm_qwen3_1.5b/final",
        base_model_path=os.path.join(os.path.expanduser("~"), "autodl-fs", "model_cache", "Qwen", "Qwen3-1.5B"),
        score_head_path="output/prm_qwen3_1.5b/final/score_head.pt",
    )

    # 测试样例
    test_response = """<think>
首先，题目给了 x + 3 = 7

移项，把 3 移到等号右边

x = 7 - 3

计算得到 x = 4
</think>
<answer>4</answer>"""

    reward = scorer.compute_reward(test_response, "4", "解方程 x + 3 = 7")
    print(json.dumps({k: v for k, v in reward.items() if k != "step_scores"}, indent=2))
