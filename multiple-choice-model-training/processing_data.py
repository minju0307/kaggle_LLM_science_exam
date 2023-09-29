from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
import json

## showing dataset function
def show_one(example):
    print(f"Question: {example['question']}")
    print(f"  A - {example['A']}")
    print(f"  B - {example['B']}")
    print(f"  C - {example['C']}")
    print(f"  D - {example['D']}")
    print(f"\nGround truth: option {example['answer']}")

## load dataset
def load_data(config):
    dataset = load_dataset('csv', data_files={'train':config['train_datasets'], 'test':config['dev_datasets']}, keep_default_na=False)
    return dataset

## preprocessing dataset
def get_encoded_datasets(dataset, config):
    tokenizer=AutoTokenizer.from_pretrained(config['model_ckpt'])
    options = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    def preprocess_function(examples):
        # Repeat each questions four times to go with the four possibilities of second sentences.
        first_sentences = [[context] * 4 for context in examples["question"]]

        # Grab all option sentences possible for each question.
        second_sentences = [[f"{examples[option][i]}" for option in options] for i in range(len(examples['question']))]

        # Flatten everything
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)

        # label idx
        encoded_labels = [options[f'{label}'] for label in examples['answer']]

        # Un-flatten
        output_dict = {k: [v[i:i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        output_dict['label'] = encoded_labels
        return output_dict

    encoded_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['question', 'A', 'B', 'C', 'D', 'answer'])
    return encoded_datasets

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in
                              features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

