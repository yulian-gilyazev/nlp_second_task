from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    evaluation_strategy: str
    eval_steps: int
    save_strategy: str
    num_labels: int
    train_classes_frequency: List[float]
    training_train_dataset_size: float
    hf_model_path: str
    device: str
