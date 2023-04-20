import json

import numpy as np
import pandas as pd
import torch
from dacite import from_dict
from torch.utils.data import Dataset
from transformers import Trainer

from config import Config


def load_config(config_path):
    with open(config_path, 'r') as file:
        config_data = json.load(file)
    return from_dict(Config, config_data)


def convert_fields_to_text(movie_name, movie_description):
    return f"Film name: {movie_name}. {movie_description}"


class FilmsDataset(Dataset):
    def __init__(self, df, transforms, train=True,
                 shuffle=True, text_converter=convert_fields_to_text):
        super().__init__()
        self.train = train
        if shuffle:
            df_data = df.iloc[np.random.permutation(len(df))]
        else:
            df_data = df.copy()
        if self.train:
            self.labels = df_data['target'].to_list()
        texts = [
            text_converter(movie_name, movie_description)
            for movie_name, movie_description in zip(df_data['movie_name'].to_list(),
                                                     df_data['movie_description'].to_list())
        ]
        self.source_texts = texts
        self.texts = transforms(texts, truncation=True, padding=True)

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.texts.items()}
        item['text'] = self.source_texts[idx]
        if self.train:
            item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


class WeightedClassesTrainer(Trainer):
    """Кастомный Trainer с весами классов."""
    def __init__(self, *args, **kwargs):
        sm = torch.nn.Softmax()
        self.classes_weights = sm(torch.sqrt(1 / torch.tensor(kwargs['classes_frequency'])))
        kwargs.pop('classes_frequency')
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.classes_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_train_val_datasets(path_to_data, transforms, train_size=1.):
    """Загрузка и подготовка тренеровочного и валидационного датасетов."""
    df = pd.read_csv(path_to_data)
    df = df.iloc[np.random.permutation(len(df))]
    threshold = int(df.shape[0] * train_size) - 1
    train_dset = FilmsDataset(df[:threshold], transforms)
    val_dset = FilmsDataset(df[threshold:], transforms)
    return train_dset, val_dset


def compute_accuracy(predictions):
    predicted_logits, gt_labels = predictions
    predicted_labels = np.argmax(predicted_logits, axis=1)
    return np.mean(predicted_labels != gt_labels)
