from typing import List, Optional, Union

from fastapi import status, HTTPException
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class HuggingFaceGenerationConfig(BaseModel):
    do_sample: bool = Field(default=False)
    early_stopping: bool = Field(default=True)
    num_beams: int = Field(default=1)
    num_return_sequences: int = Field(default=1)
    max_new_tokens: int = Field(default=32)
    min_new_tokens: int = Field(default=1)
    temperature: float = Field(default=1)
    top_p: float = Field(default=1, ge=0, le=1)
    top_k: int = Field(default=0)
    typical_p: float = Field(default=1, ge=0, le=1)
    repetition_penalty: float = Field(default=1)
    eos_token_id: Optional[int] = Field(default=None)
    pad_token_id: Optional[int] = Field(default=None)
    seed: int = Field(default=1024)

    def __hash__(self):
        return hash(
            str(self.do_sample) +
            str(self.early_stopping) +
            str(self.num_beams) +
            str(self.num_return_sequences) +
            str(self.max_new_tokens) +
            str(self.min_new_tokens) +
            str(self.temperature) +
            str(self.top_p) +
            str(self.top_k) +
            str(self.typical_p) +
            str(self.repetition_penalty) +
            str(self.eos_token_id) +
            str(self.pad_token_id) +
            str(self.seed)
        )


class HuggingFaceCompletionInputs(BaseModel):
    model: str = Field(default=...)
    prompt: str = Field(default=...)
    generation_config: HuggingFaceGenerationConfig = Field(default=HuggingFaceGenerationConfig())

    def verify_and_preprocess(self):
        # verify
        # here we only do the simplest verification for some parameters
        # we should also check with other information again in the server for:
        # - whether num_prompt_tokens exceeds model's max_seq_len, if yse we should abort request directly.
        if self.generation_config.num_return_sequences > self.generation_config.num_beams:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"num_return_sequences(get value of {self.generation_config.num_return_sequences}) "
                    f"has to less than or equal to num_beams(get value of {self.generation_config.num_beams})"
                )
            )
        if self.generation_config.min_new_tokens > self.generation_config.max_new_tokens:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"min_new_tokens(get value of {self.generation_config.min_new_tokens}) "
                    f"has to less than or equal to max_new_tokens(get value of {self.generation_config.max_new_tokens})"
                )
            )

        # make sure some parameters in generation_config is specified
        # to default values so that do_sample will not be triggered
        if not self.generation_config.do_sample:
            self.generation_config.temperature = 1.0
            self.generation_config.top_p = 1.0
            self.generation_config.top_k = 0
            self.generation_config.typical_p = 1.0


class HuggingFaceCompletionChoice(BaseModel):
    text: str = Field(default=...)
    index: int = Field(default=...)
    finish_reason: str = Field(default=...)


class HuggingFaceCompletionOutputs(BaseModel):
    choices: Optional[List[HuggingFaceCompletionChoice]] = Field(default=None)
    usage: TokenUsage = Field(default=TokenUsage())


class OpenAIChatCompletionMessage(BaseModel):
    # Note: we removed parameters relevant to function call here for the simplest use case
    role: str = Field(default=..., regex=r"(system|user|assistant)")
    content: str = Field(default=...)


class OpenAIChatCompletionInputs(BaseModel):
    model: str = Field(default=...)
    messages: List[OpenAIChatCompletionMessage] = Field(default=...)
    n: int = Field(default=1)
    max_tokens: int = Field(default=32)
    temperature: float = Field(default=1)
    top_p: float = Field(default=1, ge=0, le=1)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    presence_penalty: float = Field(default=0)
    frequency_penalty: float = Field(default=0)

    def verify_and_preprocess(self):
        # verify
        # Not do anything, you can add logit here
        pass


class OpenAIChatCompletionChoice(BaseModel):
    message: OpenAIChatCompletionMessage = Field(default=...)
    index: int = Field(default=...)
    finish_reason: str = Field(default=...)


class OpenAIChatCompletionOutputs(BaseModel):
    choices: Optional[List[OpenAIChatCompletionChoice]] = Field(default=None)
    usage: TokenUsage = Field(default=TokenUsage())


__all__ = [
    "TokenUsage",
    "HuggingFaceGenerationConfig",
    "HuggingFaceCompletionChoice",
    "HuggingFaceCompletionInputs",
    "HuggingFaceCompletionOutputs",
    "OpenAIChatCompletionMessage",
    "OpenAIChatCompletionChoice",
    "OpenAIChatCompletionInputs",
    "OpenAIChatCompletionOutputs"
]
