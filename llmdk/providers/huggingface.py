#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional, Dict
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
            model=model_name,
            token=api_key,
        )

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        completion = self._client.chat_completion(**payload)
        return completion.choices[0].message.content
