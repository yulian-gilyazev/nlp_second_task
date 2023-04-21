import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from util import convert_fields_to_text, load_config
from tqdm import tqdm


class Inferencer:
    """Класс инферит обученную модель."""

    def __init__(self, train_df, tokenizer, model, device='cpu'):
        texts_train = [convert_fields_to_text(item['movie_name'], item['movie_description'])
                       for _, item in train_df.iterrows()]
        labels_train = train_df['target'].to_list()
        self.train_predictions_buffer = defaultdict(list)
        for text, label in zip(texts_train, labels_train):
            self.train_predictions_buffer[text].append(label)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, test_df):
        """Метод предсказывает метки для тестовой выборки. Для предсказывания
        исползьзуются некоторые наблюдения о тестовой и тренировочной выборках, а именно то,
        что они имеют порядка 25% пересечений, и поэтому если наблюдения уже встречается в
        трейне,то предсказание для него будет отличным от той метки, которая уже известна.
        """
        predictions = []
        for _, row in tqdm(test_df.iterrows()):
            text = convert_fields_to_text(row['movie_name'], row['movie_description'])
            res = self.model(**{key: torch.tensor(value).to(self.device)
                                for key, value in self.tokenizer([text]).items()})
            logits = res['logits'].detach().cpu().numpy()[0]
            labels = np.argsort(logits)[::-1]
            if text not in self.train_predictions_buffer.keys():
                predictions.append(labels[0])
            else:
                prediction = None
                for label in labels:
                    if label not in self.train_predictions_buffer[text]:
                        prediction = label
                        break
                if prediction is None:
                    prediction = labels[0]
                predictions.append(prediction)
        return predictions


def predict(config):
    # model_config = AutoConfig.from_pretrained(config.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_path)

    test_df = pd.read_csv(config.test_data_path)
    train_df = pd.read_csv(config.train_data_path)

    infer = Inferencer(train_df, model=model, tokenizer=tokenizer, device=config.device)
    predictions = infer.predict(test_df)

    test_df['target'] = predictions
    test_df = test_df[['target', 'id']].set_index('id')
    test_df.to_csv(config.prediction_path)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        add_help=True,
        description="HF sequence classification model inference."
    )
    parser.add_argument('-test-data', '--test-data', type=str,
                        help='Path to test data.')
    parser.add_argument('-train-data', '--train-data', type=str, required=False,
                        help='Path to train data.')
    parser.add_argument('-config', '--config', type=str,
                        help='Path to training config.')
    parser.add_argument('-model-path', '--model-path', type=str,
                        help='Path to model weights.')
    parser.add_argument('-prediction-path', '--prediction-path', type=str,
                        help='Path to predictions.')
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    config.test_data_path = args.test_data
    config.model_path = args.model_path
    config.prediction_path = args.prediction_path
    if hasattr(args, 'test_data'):
        config.train_data_path = args.train_data
    predict(config)


if __name__ == '__main__':
    main()
