from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
import numpy as np
import json

from processing_data import DataCollatorForMultipleChoice, show_one, load_dataset, get_encoded_datasets

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

def train (args, encoded_datasets, tokenizer):
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

if __name__=='main':
    ## load config file
    with open('config.json') as f:
        config = json.load(f)

    ## load model and tokenzier
    tokenizer = AutoTokenizer.from_pretrained(config['model_ckpt'], use_fast=True)
    model = AutoModelForMultipleChoice.from_pretrained(config['model_ckpt'])

    ## load dataset
    dataset = load_dataset(config)
    encoded_datasets = get_encoded_datasets(dataset)

    ## train
    model_name = config['model_ckpt'].split("/")[-1]
    model_dir=f"{model_name}-finetuned"
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

    trainer = train(args, encoded_datasets, tokenizer)

    ## test
    evaluation_prediction = test(trainer, encoded_datasets)