#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional


class LlmInterface:
    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        self._model_name = model_name
        self._generate_kwargs = kwargs

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError
