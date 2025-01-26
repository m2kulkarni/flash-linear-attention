# preprocess_math.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import re
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer
from transformers.utils import logging
logger = logging.get_logger(__name__)
# ========== Math Processing Core ==========
DEFAULT_SYSTEM_PROMPT = """Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
Use this step-by-step format:

## Step 1: [Concise description]
[Brief explanation and calculations]

## Step 2: [Concise description]
[Brief explanation and calculations]

...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem."""

MATH_CHAT_TEMPLATE = """{% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content']|trim %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

<|start_header_id|>system<|end_header_id|>

{{ system_message }}
Cutting Knowledge Date: December 2023
Today Date: 25 Jan 2025
<|eot_id|>

{% for message in messages %}
    {% if message['role'] == 'user' %}
        <|start_header_id|>user<|end_header_id|>
        
        {{ message['content'] }}<|eot_id|>
    {% elif message['role'] == 'assistant' %}
        <|start_header_id|>assistant<|end_header_id|>
        
        {{ message['content'] }}<|eot_id|>
    {% endif %}
{% endfor %}

{% if add_generation_prompt %}
<|start_header_id|>assistant<|end_header_id|>

{% endif %}"""

def load_math_dataset(path: str, split: str) -> Dataset:
    """Load MATH dataset from directory structure"""
    data = []
    path = Path(path) / split
    for topic_dir in path.iterdir():
        if topic_dir.is_dir():
            for problem_file in topic_dir.glob("*.json"):
                with open(problem_file) as f:
                    problem = json.load(f)
                    data.append({
                        "problem": problem["problem"],
                        "solution": problem["solution"],
                        "type": problem["type"],
                        "level": problem["level"]
                    })
    return Dataset.from_list(data)

def parse_math_response(response: str) -> dict:
    """Parse MATH dataset solutions with LaTeX formatting"""
    # Extract final answer from \boxed{}
    lines = response.split("\n")
    if lines[-1].startswith('####'):
        final_answer = lines[-1].split("####")[-1].strip()
        solution = "\n".join(lines[:-1])
    else:
        # match = re.search(r"\\boxed{([^}]*)}", response)
        match = re.search(r"\\boxed{?([^}]*)}?", response)
        if not match:
            raise ValueError(f"No final answer found in boxed format: {response}")
        
        final_answer = match.group(1)
        # Remove final answer from solution text
        solution = re.sub(r"\\boxed{[^}]*}", "", response).strip()
    
    # Split solution into logical steps
    steps = []
    current_step = []
    for line in solution.split("\n"):
        line = line.strip()
        if line.startswith(("**", "For ", "Step ", "Then ", "Similarly", "Therefore")):
            if current_step:
                steps.append(" ".join(current_step))
                current_step = []
        current_step.append(line)
    
    if current_step:
        steps.append(" ".join(current_step))
    
    return {
        "reasoning": steps,
        "final_answer": final_answer,
        "step_count": len(steps)
    }

def format_math_example(example: dict, dataset_type: str) -> dict:
    """Format examples for different math datasets"""
    if dataset_type == "gsm8k":
        question = example["question"]
        answer = example["answer"]
    else:  # MATH
        question = example["problem"]
        answer = example["solution"]
    
    parsed = parse_math_response(answer)
    
    response_lines = []
    for i, step in enumerate(parsed["reasoning"]):
        if parsed["step_count"] > 2:
            response_lines.append(f"## Step {i+1}: {step}")
        else:
            response_lines.append(step)
    
    response = "\n\n".join(response_lines)
    response += f"\n\nTherefore, the final answer is: $\\boxed{{{parsed['final_answer']}}}$. I hope it is correct."

    return {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }

def tokenize_math(
    examples: Dict[str, List[Any]], 
    tokenizer: AutoTokenizer,
    seq_len: int = 2048,
    ctx_len: int = None,
    return_offsets: bool = False
) -> Dict[str, List[List[int]]]:
    """Tokenization with math-aware chunking"""
    # Combine messages into single text
    # print(examples)
    texts = tokenizer.apply_chat_template(
            examples['messages'],
            tokenize=False, 
            add_generation_prompt=False) 
    
    input_ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
    if ctx_len:
        input_ids = [seq[i:i+ctx_len] for seq in input_ids for i in range(0, len(seq), ctx_len)]

    
    return {"input_ids": input_ids}
    


def preprocess_math(
    dataset_path: str,
    dataset_type: str = "gsm8k",
    split: str = "train",
    output_dir: str = "math-data",
    tokenizer_name: str = "fla-hub/gla-1.3B-100B",
    seq_len: int = 2048,
    ctx_len: int = 1024,
    num_proc: int = 4
) -> None:
    """End-to-end preprocessing for math datasets"""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading {dataset_type} dataset...")
    if dataset_type == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=split)
    else:
        dataset = load_math_dataset(dataset_path, split)
    
    logger.info(f"Formatting {len(dataset)} examples...")
    dataset = dataset.map(
        lambda x: format_math_example(x, dataset_type),
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )

    
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = MATH_CHAT_TEMPLATE
    
    logger.info("Tokenizing dataset...")
    dataset = dataset.map(
        lambda x: tokenize_math(x, tokenizer, seq_len, ctx_len),
        batched=False,
        num_proc=num_proc,
        remove_columns=dataset.column_names
    )
    if dataset_type == "gsm8k":
        name = "main"
    else:
        name = None

    tokenized_path = f'{output_dir}/{dataset_type}/{name}/{split}' if name is not None else f'{output_dir}/{dataset_type}/{split}'
    logger.info(f"Saving processed dataset to {tokenized_path}")
    dataset.save_to_disk(tokenized_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process math datasets")
    parser.add_argument("--dataset-type", choices=["gsm8k", "math"], required=True)
    parser.add_argument("--dataset-path", default="./MATH", help="Path to MATH dataset directory")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--output-dir", default="/n/home01/mkulkarni/projects/inference-scaling/data/")
    parser.add_argument("--tokenizer", default="fla-hub/gla-1.3B-100B")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--ctx-len", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--return_offsets", default=False)
    
    args = parser.parse_args()
    
    preprocess_math(
        dataset_path=args.dataset_path,
        dataset_type=args.dataset_type,
        split=args.split,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        seq_len=args.seq_len,
        ctx_len=args.ctx_len,
        num_proc=args.num_proc
    )
