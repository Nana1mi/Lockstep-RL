import os
import argparse
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--memory-fraction", type=float, default=0.9)
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("Real GPU experiments require manual --allow-gpu")
    path = os.path.expanduser(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1,
              gpu_memory_utilization=args.memory_fraction)

    sampling_params = SamplingParams(temperature=1.0, max_tokens=args.max_tokens)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
