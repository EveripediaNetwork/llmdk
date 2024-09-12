#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional

from groq import Groq

from llmdk.providers.interface import LlmInterface


class GroqModels:
    LLAMA_3_8B_8K = 'llama3-8b-8192'
    LLAMA_3_8B = LLAMA_3_8B_8K
    LLAMA_3_1_8B_INSTANT = 'llama-3.1-8b-instant'
    LLAMA_3_1_8B = LLAMA_3_1_8B_INSTANT

    LLAMA_3_70B_8K = 'llama3-70b-8192'
    LLAMA_3_70B = LLAMA_3_70B_8K
    LLAMA_3_1_70B_VERSATILE = 'llama-3.1-70b-versatile'
    LLAMA_3_1_70B = LLAMA_3_1_70B_VERSATILE
    LLAMA_3 = LLAMA_3_1_70B_VERSATILE

    LLAMA_3_1_405B_REASONING = 'llama-3.1-405b-reasoning'

    MIXTRAL_8X_7B_32K = 'mixtral-8x7b-32768'
    MIXTRAL_8X_7B = MIXTRAL_8X_7B_32K
    MIXTRAL = MIXTRAL_8X_7B_32K

    GEMMA_7B_IT = 'gemma-7b-it'
    GEMMA_7B = GEMMA_7B_IT

    GEMMA_2_9B_IT = 'gemma2-9b-it'
    GEMMA_2_9B = GEMMA_2_9B_IT


class GroqClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('GROQ_API_KEY')

        self._client = Groq(
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

        if max_tokens is not None:
            payload['max_tokens'] = max_tokens

        message = self._client.chat.completions.create(
            **payload
        ).choices[0].message.content

        return message
