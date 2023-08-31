import asyncio
import json
import random
import sys
import time
from argparse import ArgumentParser
from os.path import abspath, dirname
from typing import List

import aiohttp
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase

sys.path.insert(0, abspath(dirname(dirname(__file__))))

from protocol.completion_task import (
    HuggingFaceGenerationConfig,
    HuggingFaceCompletionInputs,
    HuggingFaceCompletionOutputs,
)
from protocol.routes import ROUTE_POST_CONTINUOUS_BATCHING_COMPLETION, ROUTE_POST_STATIC_BATCHING_COMPLETION


def load_tokenizer(tokenizer_name_or_path: str, use_fast: bool, max_length: int) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=use_fast,
        model_max_length=max_length,
        padding_side="left",
        truncation_side="right",
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def gen_random_lens(distribution: str, len_mean: int, len_range: int, num_requests: int) -> List[int]:
    if distribution == 'uniform':
        if len_range == 0:
            return [len_mean for _ in range(num_requests)]

        low = len_mean - (len_range // 2)
        high = len_mean + (len_range // 2)
        num_to_generate = list(
            map(lambda _: random.randint(low, high), range(num_requests)))
        return num_to_generate
    elif distribution == 'exponential':
        np.random.seed(random.randint(0, 1e6))
        return [min(round(s), len_range) for s in np.random.exponential(scale=len_mean, size=num_requests)]
    elif distribution == 'capped_exponential':
        np.random.seed(random.randint(0, 1e6))
        response_lens = []
        while len(response_lens) < num_requests:
            sample = round(np.random.exponential(scale=len_mean))
            if sample <= len_range:
                response_lens.append(sample)
        return response_lens
    else:
        raise ValueError(f'unknown distribution {distribution=}')


def prepare_payloads(
    model_id: str,
    data_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_beams: int,
    prompt_len: int,
    response_lens: List[int]
) -> List[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    assert len(prompts) >= len(response_lens)

    prompts = tokenizer.batch_decode(
        tokenizer(
            prompts,
            max_length=prompt_len,
            truncation=True,
            padding="max_length"
        )["input_ids"]
    )
    prompts = random.sample(prompts, len(response_lens))
    random.shuffle(prompts)

    payloads = []
    for idx, (prompt, response_len) in enumerate(zip(prompts, response_lens)):
        payload = HuggingFaceCompletionInputs(
            model=model_id,
            prompt=prompt,
            generation_config=HuggingFaceGenerationConfig(
                max_new_tokens=response_len,
                min_new_tokens=response_len,
                num_beams=num_beams,
                num_return_sequences=1
            )
        ).dict(by_alias=True)
        payloads.append(payload)

    return payloads


async def load_test(
    payloads: List[dict],
    throughput_oriented: bool,
    batch_size: int,
    url: str
) -> dict:
    async def request_one(request_id, payload, sleep_seconds):
        await asyncio.sleep(sleep_seconds)
        request_start = time.time()
        async with aiohttp.request(
            method="post",
            url=url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60*60)
        ) as resp:
            if resp.status:
                outputs = HuggingFaceCompletionOutputs(**(await resp.json()))
            else:
                outputs = HuggingFaceCompletionOutputs()
            num_gen_tokens = outputs.usage.completion_tokens
            status = resp.status
        request_end = time.time()
        wall_time = request_end - request_start
        print(f"  - request_id={request_id} :: {status=}, {wall_time=: .4f}s, {num_gen_tokens=}")

        return num_gen_tokens, wall_time, status

    if throughput_oriented:
        total_tokens = 0
        duration = 0
        num_success = 0
        num_failed = 0
        for start_idx in range(0, len(payloads), batch_size):
            end_idx = start_idx + batch_size
            start = time.time()
            batch_results = await asyncio.gather(
                *[
                    request_one(start_idx + idx, payload, 0)
                    for idx, payload in enumerate(payloads[start_idx: end_idx])
                ]
            )
            end = time.time()
            duration += (end - start)
            total_tokens += sum([each[0] for each in batch_results if each[2] == 200])
            if all(each[2] == 200 for each in batch_results):
                num_success += 1
            else:
                num_failed += 1
        latency = -1
        throughput = total_tokens / duration
        fail_rate = f"{num_failed / (num_failed + num_success) * 100: .4f}%"
    else:
        sleep_sec = 0
        tasks = []
        for start_idx in range(0, len(payloads), batch_size):
            end_idx = start_idx + batch_size
            sleep_sec += (0 if throughput_oriented else 1)
            tasks += [
                asyncio.create_task(request_one(start_idx + idx, payload, sleep_sec))
                for idx, payload in enumerate(payloads[start_idx: end_idx])
            ]
        results = await asyncio.gather(*tasks)
        results = [each for each in results if each[2] == 200]
        latency = pd.Series([each[1] for each in results]).describe().to_dict()
        throughput = -1
        fail_rate = f"{(len(payloads) - len(results)) / len(payloads) * 100: .4f}%"

    return {
        "throughput_oriented": throughput_oriented,
        "batch_size": batch_size,
        "latency(s)": latency,
        "throughput(generated_tokens/s)": throughput,
        "fail_rate": fail_rate
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_id", type=str, default="")  # default to not specify if request to server directly
    parser.add_argument("--data_path", type=str, default="load_test_data.json")
    parser.add_argument("--save_path", type=str, default="load_test_report.json")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="gpt2")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--throughput_oriented", action="store_true")
    # when throughput_oriented is True, num_beams will be treated as batch size
    # which means how many requests will be sent to inference server at the same time,
    # where each request's num_beams is 1
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=512)
    parser.add_argument("--response_len_mean", type=int, default=512)
    parser.add_argument("--response_len_range", type=int, default=0)
    parser.add_argument("--distribution", type=str, choices=["uniform", "exponential", "capped_exponential"], default="uniform")
    parser.add_argument("--num_emit", type=int, default=60)  # num batches or num seconds continues
    parser.add_argument("--qps", type=int, default=1)  # used only when throughput_oriented is not True
    parser.add_argument("--server_type", type=str, default="cb", choices=["cb", "sb"])
    args = parser.parse_args()

    if args.distribution == "uniform":
        assert args.response_len_mean > args.response_len_range
    else:
        assert args.response_len_mean <= args.response_len_range

    tokenizer = load_tokenizer(args.tokenizer_name_or_path, args.use_fast_tokenizer, args.prompt_len)

    if args.throughput_oriented:
        num_requests = args.num_beams * args.num_emit
        response_lens = [args.response_len_mean for _ in range(num_requests)]
        args.num_beams = 1
    else:
        num_requests = args.qps * args.num_emit
        response_lens = gen_random_lens(args.distribution, args.response_len_mean, args.response_len_range, num_requests)
    batch_size = num_requests // args.num_emit

    payloads = prepare_payloads(args.model_id, args.data_path, tokenizer, args.num_beams, args.prompt_len, response_lens)

    print(
        f"Load Test :: {args.throughput_oriented=}, {num_requests=}, {batch_size=}, "
        f"{args.response_len_mean=}, {args.response_len_range=}, {args.distribution=}"
    )

    route = ROUTE_POST_CONTINUOUS_BATCHING_COMPLETION if args.server_type == "cb" else ROUTE_POST_STATIC_BATCHING_COMPLETION
    report = asyncio.run(
        load_test(
            payloads, args.throughput_oriented, batch_size, f"{args.host}:{args.port}{route}"
        )
    )
    report["response_len_mean"] = args.response_len_mean
    report["response_len_range"] = args.response_len_range
    report["distribution"] = args.distribution

    print("REPORT ::")
    for k, v in report.items():
        print(f"  - {k}: {v}")

    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
