virtualenv env
source env/bin/activate
pip install -r requirements.txt
python3 src/inference.py --train-data data/train.csv --test-data data/test.csv --config configs/config.json --model-path models/ditilroberta-base-finetuned
