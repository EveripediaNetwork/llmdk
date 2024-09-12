#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional

from anthropic import Anthropic

from llmdk.providers.interface import LlmInterface


class AnthropicClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('ANTHROPIC_API_KEY')

        self._client = Anthropic(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            'model': self._model_name,
            'messages': [{
                "role": "user",
                "content": prompt,
            }],
        }

        if temperature is not None:
            payload['temperature'] = temperature

        # Required by Anthropic
        if max_tokens is None:
            max_tokens = 4096
        payload['max_tokens'] = max_tokens

        message = self._client.messages.create(**payload).content[0].text
        return message
