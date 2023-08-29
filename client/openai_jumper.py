import os
import time
from logging import getLogger, Logger
from typing import Optional, Tuple
from threading import Thread

import openai
from openai.util import convert_to_dict
from pydantic import BaseModel, Field, Required

from protocol.completion_task import (
    TokenUsage,
    OpenAIChatCompletionMessage,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionInputs,
    OpenAIChatCompletionOutputs
)
from protocol.error import Error


AVAILABLE_OPENAI_CHAT_COMPLETION_MODELS = [
    "gpt-3.5-turbo",
    # You can extend available chat completion models here by yourself
]


class OpenAIJumperConfig(BaseModel):
    api_key: str = Field(default=Required)
    org_id: Optional[str] = Field(default=None)


class OpenAIJumper:
    def __init__(self, config: OpenAIJumperConfig, logger: Optional[Logger] = None):
        self.config = config
        self.logger = logger if logger else getLogger(__name__)

        self.workload = 0
        self.referenced = 0
        self._available = True

    async def chat_completion(
        self,
        inputs: OpenAIChatCompletionInputs,
        max_retries: int = 3
    ) -> Tuple[
        OpenAIChatCompletionOutputs,
        Optional[Error],
        int
    ]:
        if inputs.model not in AVAILABLE_OPENAI_CHAT_COMPLETION_MODELS:
            error_body = Error(
                type="ValueError",
                detail=f"LookupError: Required model [{inputs.model}] is not available, "
                       f"available models are {AVAILABLE_OPENAI_CHAT_COMPLETION_MODELS}"
            )
            return OpenAIChatCompletionOutputs(), error_body, 404

        request_dict = inputs.dict(exclude_none=True)
        request_dict.update({"api_key": self.config.api_key})
        if self.config.org_id:
            request_dict.update({"organization": self.config.org_id})

        try:
            self.referenced += 1
            resp = await openai.ChatCompletion.acreate(**request_dict)
        except openai.error.Timeout as e:
            self.referenced -= 1
            if max_retries > 0:
                max_retries -= 1
                self.logger.warning(
                    msg=f"Request to openai chat completion api timeout, will retry again (chance_left={max_retries})"
                )
                return await self.chat_completion(inputs, max_retries=max_retries)
            error_body = Error(
                type=e.__class__.__name__,
                detail=e.user_message
            )
            self.logger.error(msg=str(error_body))
            return OpenAIChatCompletionOutputs(), error_body, e.http_status
        except openai.error.OpenAIError as e:
            self.referenced -= 1
            error_body = Error(
                type=e.__class__.__name__,
                detail=e.user_message
            )
            self.logger.error(msg=str(error_body))
            return OpenAIChatCompletionOutputs(), error_body, e.http_status
        except Exception as e:
            self.referenced -= 1
            error_body = Error(
                type=e.__class__.__name__,
                detail=str(e)
            )
            self.logger.error(msg=str(error_body))
            return OpenAIChatCompletionOutputs(), error_body, 500
        else:
            self.referenced -= 1
            resp = convert_to_dict(resp)
            outputs = OpenAIChatCompletionOutputs(
                choices=[
                    OpenAIChatCompletionChoice(
                        message=OpenAIChatCompletionMessage(
                            role=choice["message"]["role"],
                            content=choice["message"]["content"]
                        ),
                        index=choice["index"],
                        finish_reason=choice["finish_reason"]
                    ) for choice in resp["choices"]
                ],
                usage=TokenUsage(**resp["usage"])
            )

            self.workload += outputs.usage.total_tokens

            return outputs, None, 200

    def reset_workload(self):
        self.workload = 0

    def freeze(self):
        self._available = False

    @property
    def available(self):
        return self._available

    def __eq__(self, other: "OpenAIJumper"):
        return self.workload == other.workload

    def __gt__(self, other: "OpenAIJumper"):
        return self.workload > other.workload

    def __ge__(self, other: "OpenAIJumper"):
        return self.workload >= other.workload

    def __lt__(self, other: "OpenAIJumper"):
        return self.workload < other.workload

    def __le__(self, other: "OpenAIJumper"):
        return self.workload <= other.workload

    def __hash__(self):
        return hash(self.config.api_key)


def destroy_openai_jumper(openai_jumper: OpenAIJumper):
    def destroy():
        openai_jumper.freeze()
        while True:
            if openai_jumper.referenced == 0:
                del openai_jumper
                return
            time.sleep(1)

    Thread(target=destroy, daemon=True).start()


__all__ = [
    "AVAILABLE_OPENAI_CHAT_COMPLETION_MODELS",
    "OpenAIJumper",
    "OpenAIJumperConfig",
    "destroy_openai_jumper",
]
