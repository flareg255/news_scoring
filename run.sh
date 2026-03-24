#!/bin/bash

# 仮想環境を有効化してmain.pyを実行する

cd "$(dirname "$0")"

source .venv/bin/activate

python main.py
