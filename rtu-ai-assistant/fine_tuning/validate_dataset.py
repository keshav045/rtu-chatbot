"""
Validation script for RTU fine-tuning datasets.
Checks JSON formatting, ChatML roles, token distributions, and task balances.
"""

import json
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent / "rtu_train_dataset_expanded.jsonl"

def validate():
    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        return False

    total = 0
    errors = 0
    task_counts = {}
    system_prompts = set()
    role_sets = set()
    char_lengths = []

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Line {idx}: JSON parse error: {e}")
                errors += 1
                continue

            task = obj.get("task", "missing")
            task_counts[task] = task_counts.get(task, 0) + 1

            msgs = obj.get("messages", [])
            if not msgs or len(msgs) < 3:
                print(f"Line {idx}: Invalid messages list length ({len(msgs)})")
                errors += 1
                continue

            roles = tuple(m.get("role") for m in msgs)
            role_sets.add(roles)

            sys_content = msgs[0].get("content", "")
            system_prompts.add(sys_content)

            total_chars = sum(len(m.get("content", "")) for m in msgs)
            char_lengths.append(total_chars)

    print("=" * 60)
    print("RTU Fine-Tuning Dataset Validation Report")
    print("=" * 60)
    print(f"Dataset Path        : {DATASET_PATH}")
    print(f"Total Examples      : {total}")
    print(f"Parse Errors        : {errors}")
    print(f"Distinct Role Layout: {role_sets}")
    print(f"System Prompts Count: {len(system_prompts)}")
    print(f"Average Example Size: {sum(char_lengths) / len(char_lengths):.0f} characters")
    print(f"Max Example Size    : {max(char_lengths)} characters")
    print(f"Min Example Size    : {min(char_lengths)} characters")
    print("\nTask Breakdown:")
    for task, count in sorted(task_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {task:<25}: {count:>4} ({count / total * 100:.1f}%)")

    success = (errors == 0 and total >= 500 and len(system_prompts) == 1 and role_sets == {('system', 'user', 'assistant')})
    print("\nResult:", "VALIDATION PASSED (100% HEALTHY)" if success else "VALIDATION FAILED")
    return success

if __name__ == "__main__":
    validate()
