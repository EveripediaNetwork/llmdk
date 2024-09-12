#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional

from openai import OpenAI

from llmdk.providers.interface import LlmInterface


class OpenAiClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('OPENAI_API_KEY') or 'EMPTY'

        if not base_url:
            base_url = env.get('OPENAI_API_URL') or None

        self._client = OpenAI(
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

        max_tokens = max_tokens or self._max_tokens
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens

        completion = self._client.chat.completions.create(**payload)
        message = completion.choices[0].message.content
        return message
