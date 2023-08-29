# Adopt from Hugging Face text-generation-inference

from typing import *

import torch
from pydantic import BaseModel, Field, Required

from .logits_process import (
    HeterogeneousRepetitionPenaltyLogitsProcessor,
    HeterogeneousTemperatureLogitsWarper,
    HeterogeneousTopKLogitsWarper,
    HeterogeneousTopPLogitsWarper,
    HeterogeneousTypicalLogitsWarper,
    HeterogeneousProcessorWrapper,
)


class NextTokensChooserOutput(BaseModel):
    next_probs: List[float] = Field(default=Required)
    next_token_ids: List[int] = Field(default=Required)


class HeterogeneousNextTokensChooser:
    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        temperature: List[float],
        repetition_penalty: List[float],
        top_k: List[int],
        top_p: List[float],
        typical_p: List[float],
        do_sample: List[bool],
        num_beams: List[int],
        seeds: List[int],
    ):
        warpers = []

        self.repetition_processor = (
            HeterogeneousRepetitionPenaltyLogitsProcessor(
                repetition_penalty, dtype, device
            )
            if any([x != 1.0 for x in repetition_penalty])
            else None
        )

        if any([x != 1.0 for x in temperature]):
            do_sample = [
                sample or x != 1.0 for x, sample in zip(temperature, do_sample)
            ]
            warpers.append(
                HeterogeneousTemperatureLogitsWarper(temperature, dtype, device)
            )

        if any([x != 0 for x in top_k]):
            do_sample = [sample or x != 0 for x, sample in zip(top_k, do_sample)]
            warpers.append(HeterogeneousTopKLogitsWarper(top_k, device))

        if any([x < 1.0 for x in top_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(top_p, do_sample)]
            warpers.append(HeterogeneousTopPLogitsWarper(top_p, dtype, device))

        if any([x < 1.0 for x in typical_p]):
            do_sample = [sample or x < 1.0 for x, sample in zip(typical_p, do_sample)]
            warpers.append(HeterogeneousTypicalLogitsWarper(typical_p, dtype, device))

        self.warpers = warpers

        if any(do_sample):
            self.choice = HeterogeneousSampling(do_sample, num_beams, seeds, device)
        else:
            self.choice = Greedy(num_beams)

        self.seeds = seeds
        self.do_sample = do_sample
        self.dtype = dtype
        self.device = device

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> List[NextTokensChooserOutput]:
        if self.repetition_processor is not None:
            scores = self.repetition_processor(input_ids, scores)

        for warper in self.warpers:
            scores = warper(input_ids, scores)

        log_scores = torch.log_softmax(scores, dim=-1)

        token_ids = self.choice(scores)
        log_probs = [
            [log_scores[i, token_id].item() for token_id in beam_nxt_token_ids]
            for i, beam_nxt_token_ids in enumerate(token_ids)
        ]

        return [
            NextTokensChooserOutput(next_token_ids=nxt_token_ids, next_probs=nxt_probs)
            for nxt_token_ids, nxt_probs in zip(token_ids, log_probs)
        ]


class Sampling:
    def __init__(self, num_beams: int, seed: int, device: str = "cpu"):
        self.num_beams = num_beams
        self.generator = torch.Generator(device)
        self.generator.manual_seed(seed)
        self.seed = seed

    def __call__(self, logits) -> List[int]:
        probs = torch.nn.functional.softmax(logits, -1)
        # Avoid GPU<->CPU sync done by torch multinomial
        # See: https://github.com/pytorch/pytorch/blob/925a3788ec5c06db62ca732a0e9425a26a00916f/aten/src/ATen/native/Distributions.cpp#L631-L637
        q = torch.empty_like(probs).exponential_(1, generator=self.generator)
        return torch.topk(probs.div_(q), k=self.num_beams + 1).indices.tolist()


class Greedy:
    def __init__(self, num_beams: List[int]):
        self.num_beams = num_beams

    def __call__(self, logits) -> List[List[int]]:
        return torch.topk(logits, k=max(self.num_beams) + 1, dim=-1).indices.tolist()


class HeterogeneousSampling:
    r"""
    Mixed greedy and probabilistic sampling. Compute both and pick the right one for each sample.
    """

    def __init__(self, do_sample: List[bool], num_beams: List[int], seeds: List[int], device: torch.device):
        self.num_beams = num_beams
        self.seeds = seeds

        greedy_indices = []
        self.sampling_mapping = {}
        for i, (sample, seed) in enumerate(zip(do_sample, seeds)):
            if sample:
                self.sampling_mapping[i] = Sampling(num_beams[i], seed, device)
            else:
                greedy_indices.append(i)

        self.greedy_indices = greedy_indices

    def __call__(self, logits) -> List[List[int]]:
        out = [None for _ in range(logits.shape[0])]
        if self.greedy_indices:
            # Computing for all indices is faster than slicing
            greedy = Greedy(self.num_beams)
            out = greedy(logits)

        for i, sampling in self.sampling_mapping.items():
            out[i] = sampling(logits[i])
        return out
