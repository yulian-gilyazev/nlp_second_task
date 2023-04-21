import argparse
import logging

from transformers import AutoModelForSequenceClassification, \
    AutoTokenizer, TrainingArguments

from util import WeightedClassesTrainer, compute_accuracy, \
    load_config, load_train_val_datasets


def train(config):
    model = AutoModelForSequenceClassification.from_pretrained(config.hf_model_path,
                                                               num_labels=config.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(config.hf_model_path)
    train_dataset, val_dataset = load_train_val_datasets(config.train_data_path, 
          tokenizer, train_size=config.training_train_dataset_size)
    training_args = TrainingArguments(
        output_dir=config.model_path,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        no_cuda=(config.device == 'cpu')
    )
    trainer = WeightedClassesTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_accuracy,
        classes_frequency=config.train_classes_frequency,
        device=config.device
    )
    trainer.train()
    trainer.save_model(config.model_path)
    logging.info(f'Model state saved in {config.model_path}!')


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        add_help=True,
        description="HF sequence classification model training."
    )
    parser.add_argument('-train-data', '--train-data', type=str, required=False,
                        help='Path to train data.')
    parser.add_argument('-config', '--config', type=str,
                        help='Path to training config.')
    parser.add_argument('-model-path', '--model-path', type=str,
                        help='Path to model weights.')
    args = parser.parse_args(args)
    return args


def main():
    args = parse_args()
    config = load_config(args.config)
    config.model_path = args.model_path
    if hasattr(args, 'train_data'):
        config.train_data_path = args.train_data
    train(config)


if __name__ == '__main__':
    main()

