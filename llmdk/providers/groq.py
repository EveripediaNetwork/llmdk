#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional, Dict
from groq import Groq
from llmdk.providers.interface import LlmInterface


class GroqClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not api_key:
            api_key = env.get('GROQ_API_KEY')

        self._client = Groq(
            api_key=api_key,
        )

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        completion = self._client.chat.completions.create(**payload)
        return completion.choices[0].message.content
