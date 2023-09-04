#!/bin/bash
poetry run python -m build --wheel
rm -r build
