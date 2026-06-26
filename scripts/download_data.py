"""
从 ModelScope 下载数学推理数据集（MathR + R1 Distill）
产出: data/train_math_all.jsonl（约 4 万条）
"""
import json, os, re, sys

def clean_answer(answer):
    ans = answer.strip()
    boxed = re.findall(r'\\boxed\{([^}]+)\}', ans)
    if boxed:
        return boxed[-1].strip()
    boxed2 = re.findall(r'\\\(\s*\\boxed\{([^}]+)\}', ans)
    if boxed2:
        return boxed2[-1].strip()
    ans = re.sub(r'^[因此所以故综上答案最终答案ThusThereforeHenceSoThe answer is]+[，,:.：\s]*', '', ans, flags=re.IGNORECASE).strip()
    ans = ans.replace(r'\(', '').replace(r'\)', '').replace(r'\[', '').replace(r'\]', '')
    if len(ans) > 100:
        nums = re.findall(r'[-]?\d+\.?\d*', ans)
        if nums:
            ans = nums[-1]
    return ans.strip()


def download_mathr(out_dir):
    """下载 MathR（AMC/AIME 竞赛题，约 4000 条）"""
    from modelscope.msdatasets import MsDataset
    print("下载 MathR...")
    ds = MsDataset.load('modelscope/MathR', subset_name='default', split='train')
    count = 0
    path = f'{out_dir}/train_math.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for sample in ds:
            msgs = sample['messages']
            user_msg = next((m for m in msgs if m['role'] == 'user'), None)
            assistant_msg = next((m for m in msgs if m['role'] == 'assistant'), None)
            if not user_msg or not assistant_msg:
                continue
            instruction = user_msg['content']
            response = assistant_msg['content']
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            reasoning = think_match.group(1).strip() if think_match else ''
            answer = answer_match.group(1).strip() if answer_match else ''
            item = {'instruction': instruction, 'reasoning': reasoning,
                    'answer': clean_answer(answer), 'source': sample['source']}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'  MathR: {count} 条 -> {path}')
    return count


def download_r1_math(out_dir):
    """下载 R1 Distill 数学子集（约 3.6 万条）"""
    from modelscope.msdatasets import MsDataset
    print("下载 Chinese-DeepSeek-R1-Distill-data-110k（仅数学子集）...")
    ds = MsDataset.load('liucong/Chinese-DeepSeek-R1-Distill-data-110k', split='train')
    math_repos = {'EduChat-Math', 'meta-math/GSM8K_zh',
                  'gavinluo/applied_math', 'Haijian/Advanced-Math'}
    count = 0
    path = f'{out_dir}/train_math_r1.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for s in ds:
            if s['repo_name'] not in math_repos:
                continue
            instruction = s['input']
            reasoning = s.get('reasoning_content', '') or ''
            content = s.get('content', '') or ''
            boxed = re.findall(r'\\boxed\{([^}]+)\}', content)
            answer = boxed[-1] if boxed else ''
            if not instruction or not reasoning:
                continue
            item = {'instruction': instruction, 'reasoning': reasoning,
                    'answer': clean_answer(answer), 'source': s['repo_name']}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            count += 1
    print(f'  R1 Math: {count} 条 -> {path}')
    return count


def merge_all(out_dir):
    """合并所有数据到 train_math_all.jsonl"""
    total = 0
    output = f'{out_dir}/train_math_all.jsonl'
    with open(output, 'w', encoding='utf-8') as out:
        for fname in ['train_math.jsonl', 'train_math_r1.jsonl']:
            path = f'{out_dir}/{fname}'
            if os.path.exists(path):
                with open(path) as inf:
                    for line in inf:
                        if line.strip():
                            out.write(line)
                            total += 1
    print(f'\n合并完成: {total} 条 -> {output}')


if __name__ == '__main__':
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    os.makedirs(out_dir, exist_ok=True)
    n1 = download_mathr(out_dir)
    n2 = download_r1_math(out_dir)
    merge_all(out_dir)
    print(f'总计: {n1 + n2} 条训练数据')
