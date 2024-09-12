#!/bin/bash
set -e
cd $(dirname $0)
rm -rf dist/*.whl
pip install -r requirements-dev.txt
pip wheel --wheel-dir=./dist --no-deps .
twine upload dist/*.whl
