#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import environ as env
from typing import Any, Optional, Dict
from ollama import Client
from llmdk.providers.interface import LlmInterface


class OllamaClient(LlmInterface):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        headers: Optional[dict] = None,
        options: Optional[dict] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name=model_name, **kwargs)

        if not base_url:
            base_url = env.get('OLLAMA_API_URL') or 'http://localhost:11434'

        self._client = Client(
            host=base_url,
            headers=headers or {},
        )

        self._options = options or {}

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        payload['options'] = self._options
        response = self._client.chat(**payload)
        return response['message']['content']
