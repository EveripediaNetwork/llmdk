#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

import llmdk

setup(
    name='llmdk',
    version=llmdk.__version__,
    description='LLM Development Kit for common APIs',
    url='https://github.com/EveripediaNetwork/llmdk',
    author='Rodrigo Martínez (brunneis)',
    author_email='dev@brunneis.com',
    license='GNU General Public License v3 (GPLv3)',
    packages=find_packages(),
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        'requests',
        'openai',
        'anthropic',
        'groq',
        'ollama',
        'huggingface-hub',
    ],
    package_data={
        'llmdk': [
            '',
        ],
    },
)
