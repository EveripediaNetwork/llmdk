#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional

from ollama import Client

from llmdk.providers.interface import LlmInterface


class OllamaClient(LlmInterface):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        headers: Optional[dict] = None,
        options: Optional[dict] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        self._client = Client(
            host=base_url,
            headers=headers,
        )

        self._options = options
        if self._options is None:
            self._options = {}

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

        options = dict(self._options)
        payload['options'] = options

        if temperature is not None:
            options['temperature'] = temperature

        if max_tokens is not None:
            options['num_predict'] = max_tokens

        message = self._client.chat(
            **payload
        )['message']['content']

        return message
