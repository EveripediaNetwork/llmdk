#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional


class LlmInterface:
    def __init__(
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
    ):
        self._model_name = model_name
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError
