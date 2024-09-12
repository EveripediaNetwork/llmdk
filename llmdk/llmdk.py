#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

from llmdk.providers.anthropic import AnthropicClient
from llmdk.providers.groq import GroqClient
from llmdk.providers.huggingface import HuggingFaceClient
from llmdk.providers.ollama import OllamaClient
from llmdk.providers.openai import OpenAiClient
from llmdk.providers.vllm import VllmClient


class Providers(Enum):
    ANTHROPIC = 'anthropic'
    GROQ = 'groq'
    HUGGINGFACE = 'huggingface'
    OLLAMA = 'ollama'
    OPENAI = 'openai'
    VLLM = 'vllm'


class Llmdk:
    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str = None,
        base_url: str = None,
    ):
        self._client = None

        if (
            provider == Providers.ANTHROPIC
            or provider == Providers.ANTHROPIC.value
        ):
            self._client = AnthropicClient(
                api_key=api_key,
                model_name=model_name,
            )
            return

        if (
            provider == Providers.GROQ
            or provider == Providers.GROQ.value
        ):
            self._client = GroqClient(
                api_key=api_key,
                model_name=model_name,
            )
            return

        if (
            provider == Providers.HUGGINGFACE
            or provider == Providers.HUGGINGFACE.value
        ):
            self._client = HuggingFaceClient(
                api_key=api_key,
                model_name=model_name,
            )
            return

        if (
            provider == Providers.OLLAMA
            or provider == Providers.OLLAMA.value
        ):
            self._client = OllamaClient(
                base_url=base_url,
                model_name=model_name,
            )
            return

        if (
            provider == Providers.OPENAI
            or provider == Providers.OPENAI.value
        ):
            self._client = OpenAiClient(
                api_key=api_key,
                model_name=model_name,
            )
            return

        if (
            provider == Providers.VLLM
            or provider == Providers.VLLM.value
        ):
            self._client = VllmClient(
                base_url=base_url,
            )
            return

        raise ValueError(f"Provider {provider} is not supported")

    # Fallback to the original client
    def __getattr__(self, name):
        return getattr(self._client, name)
