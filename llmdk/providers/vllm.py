#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict
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

    def _execute_request(self, payload: Dict[str, Any]) -> str:
        response = self._post(payload)
        return response['choices'][0]['message']['content']
