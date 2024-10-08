#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Iterator, Optional, Dict
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

    def _execute_stream_request(
        self,
        payload: Dict[str, Any],
    ) -> Iterator[str]:
        payload['stream'] = True
        for chunk in self._client.chat.completions.create(**payload):
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
