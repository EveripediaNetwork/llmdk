#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional

from openai import OpenAI

from llmdk.providers.interface import LlmInterface


class OpenAiModels:
    GPT_4O_2024_05_13 = 'gpt-4o-2024-05-13'
    GPT_4O_MINI_2024_07_18 = 'gpt-4o-mini-2024-07-18'
    GPT_4O_MINI = 'gpt-4o-mini'
    GPT_4O_2024_08_06 = 'gpt-4o-2024-08-06'
    CHATGPT_40_LATEST = 'chatgpt-4o-latest'
    GPT_4O = GPT_4O_2024_08_06

    GPT_4_TURBO_2024_04_09 = 'gpt-4-turbo-2024-04-09'
    GPT_4_2024_04_09 = GPT_4_TURBO_2024_04_09
    GPT_4_TURBO = 'gpt-4-turbo'
    GPT_4 = GPT_4_TURBO_2024_04_09

    GPT_4_TURBO_0125_128K = 'gpt-4-0125-preview'
    GPT_4_0125_128K = GPT_4_TURBO_0125_128K
    GPT_4_0613_8K = 'gpt-4-0613'
    GPT_4_0613_32K = 'gpt-4-32k-0613'
    GPT_4_TURBO_1106_128K = 'gpt-4-1106-preview'
    GPT_4_0314_8K = 'gpt-4-0314'
    GPT_4_0314_32K_LEGACY = 'gpt-4-32k-0314'

    GPT_3_5_TURBO_0125_16K = 'gpt-3.5-turbo-0125'
    GPT_3_5_TURBO_1106_16K = 'gpt-3.5-turbo-1106'
    GPT_3_5_TURBO = 'gpt-3.5-turbo'
    GPT_3_5 = GPT_3_5_TURBO


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
