from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
import numpy as np
import json
import argparse

from processing_data import DataCollatorForMultipleChoice, show_one, load_data, get_encoded_datasets

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

def train (args, encoded_datasets, tokenizer, model):
    trainer = Trainer(model,
            args,
            train_dataset=encoded_datasets["train"],
            eval_dataset=encoded_datasets["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer),
            compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer

def test(trainer, encoded_datasets):
    predictions = trainer.predict(encoded_datasets["test"])
    return predictions


def main(cli_args):
    ## load config file
    with open(f'{cli_args.config}', 'r', encoding='utf-8') as f:
        config = json.load(f)
    # print(config)

    ## load model and tokenzier
    tokenizer = AutoTokenizer.from_pretrained(config['model_ckpt'])
    model = AutoModelForMultipleChoice.from_pretrained(config['model_ckpt'])

    # print(tokenizer)
    # print(model)

    ## load dataset
    dataset = load_data(config)
    encoded_datasets = get_encoded_datasets(dataset, config)
    print(encoded_datasets["train"][0])
    
    ## train
    model_name = config['model_ckpt'].split("/")[-1]
    model_dir=f"{cli_args.output_dir}"
    args = TrainingArguments(
        output_dir = model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['num_train_epoch'],
        weight_decay=config['weight_decay'],
        report_to='none'
    )

    trainer = train(args, encoded_datasets, tokenizer, model)

    ## test
    evaluation_prediction = test(trainer, encoded_datasets)


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--config", type=str, required=True)
    cli_parser.add_argument("--output_dir", type=str, required=True)
    cli_args = cli_parser.parse_args()
    main(cli_args)