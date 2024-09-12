#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional

import requests

from llmdk.providers.interface import LlmInterface


class VllmClient(LlmInterface):
    def __init__(
        self,
        base_url: str,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._base_url = base_url

    def _post(self, payload: Dict) -> Dict:
        response = requests.post(
            self._base_url,
            json={
                **payload,
                'stream': False,
            },
        )
        return response.json()

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "prompt": prompt,
        }

        if temperature is not None:
            payload['temperature'] = temperature

        if max_tokens is not None:
            payload['max_tokens'] = max_tokens

        message = self._post(payload)['text'][0]

        return message
