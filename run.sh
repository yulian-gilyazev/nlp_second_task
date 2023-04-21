#!/bin/bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt

python3 src/train.py --train-data data/train.csv --config configs/config.json --model-path models/distilroberta-base-finetuned
python3 src/inference.py --train-data data/train.csv --test-data data/test.csv --prediction-path data/submission.txt --config configs/config.json --model-path models/distilroberta-base-finetuned/checkpoint-112