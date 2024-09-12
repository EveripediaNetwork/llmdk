#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional

from huggingface_hub import InferenceClient

from llmdk.providers.interface import LlmInterface


class HuggingFaceClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('HF_TOKEN')

        self._client = InferenceClient(
            model_name,
            token=api_key,
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            'messages': [{
                "role": "user",
                "content": prompt,
            }],
        }

        if temperature is not None:
            payload['temperature'] = temperature

        max_tokens = max_tokens or self._max_tokens
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens

        completion = self._client.chat_completion(**payload)
        message = completion.choices[0].message.content
        return message
