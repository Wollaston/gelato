from pathlib import Path
from typing import Any, Optional

import evaluate
import numpy as np
import wandb
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from pydantic import BaseModel
from seqscore import conll
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


class FineTune(BaseModel):
    train_path: Path
    test_path: Path
    model: str
    wandb_project: str
    sweeps: int
    output_dir: Path
    id2label: Optional[dict[int, str]] = None
    label2id: Optional[dict[str, int]] = None

    def get_dataset(self) -> DatasetDict:
        def ingest_conll(path: Path) -> tuple[list[list[str]], list[list[str]]]:
            with open(path) as file:
                all_tokens: list[list[str]] = []
                all_labels: list[list[str]] = []

                mention_encoding = conll.get_encoding("BIO")
                ingester = conll.CoNLLIngester(mention_encoding)

                docs = ingester.ingest(file, str(path), None)
                for doc in docs:
                    for seq in doc:
                        all_tokens.append(list(seq.tokens))
                        all_labels.append(list(seq.labels))

            return (all_tokens, all_labels)

        train_tokens, train_labels = ingest_conll(self.train_path)
        test_tokens, test_labels = ingest_conll(self.test_path)

        labels_set: set[str] = set()
        [labels_set.update(labels) for labels in train_labels]
        unique_labels: list[str] = sorted(list(labels_set))
        print(f"Unique tags found: {unique_labels}")

        features = Features(
            {
                "tokens": Sequence(Value("string")),
                "ner_tags": Sequence(ClassLabel(names=unique_labels)),
            }
        )

        train_dataset = Dataset.from_dict(
            {"tokens": train_tokens, "ner_tags": train_labels}, features=features
        )
        test_dataset = Dataset.from_dict(
            {"tokens": test_tokens, "ner_tags": test_labels}, features=features
        )

        return DatasetDict({"train": train_dataset, "test": test_dataset})

    def tokenize_and_align_labels(
        self, examples: dict[str, Any], tokenizer: PreTrainedTokenizerBase
    ):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def get_sweep_config(self) -> dict[str, Any]:
        return {
            "method": "bayes",  # "grid", "random" or "bayes"
            "metric": {
                "name": "eval/f1",
                "goal": "maximize",
            },
            "parameters": {
                "learning_rate": {
                    "values": [
                        0.001,
                        0.003,
                        0.005,
                        0.0001,
                        0.0003,
                        0.0005,
                        0.00001,
                        0.00003,
                        0.00005,
                        0.000001,
                        0.000003,
                        0.000005,
                    ],
                },
                "per_device_train_batch_size": {
                    "values": [1, 2, 4, 8, 16, 32, 64, 128],
                },
                "num_train_epochs": {
                    "min": 1,
                    "max": 50,
                    "distribution": "int_uniform",
                },
                "weight_decay": {
                    "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                },
                "warmup_ratio": {
                    "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                },
            },
        }

    def train(self) -> None:
        wandb.init(project=self.wandb_project)

        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model, add_prefix_space=True
        )

        dataset = self.get_dataset()
        tokenized_datasets = {}

        for split in dataset:
            tokenized_datasets[split] = dataset[split].map(
                self.tokenize_and_align_labels,
                batched=True,
                remove_columns=dataset[split].column_names,
                fn_kwargs={"tokenizer": tokenizer},
            )

        label_list = dataset["train"].features["ner_tags"].feature.names
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.label2id = {label: i for i, label in enumerate(label_list)}

        model = AutoModelForTokenClassification.from_pretrained(
            self.model,
            num_labels=len(label_list),
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        hyperparams = wandb.config

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=hyperparams.learning_rate,
            per_device_train_batch_size=hyperparams.per_device_train_batch_size,
            per_device_eval_batch_size=hyperparams.per_device_train_batch_size,
            num_train_epochs=hyperparams.num_train_epochs,
            weight_decay=hyperparams.weight_decay,
            warmup_ratio=hyperparams.warmup_ratio,
            metric_for_best_model="f1",
            report_to="wandb",
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate(tokenized_datasets["test"])
        wandb.log(metrics)

    def sweep(self) -> None:
        sweep_id = wandb.sweep(self.get_sweep_config(), project=self.wandb_project)
        wandb.agent(
            sweep_id=sweep_id,
            function=self.train,
            project=self.wandb_project,
            count=self.sweeps,
        )

    def compute_metrics(self, pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)

        seqeval = evaluate.load("seqeval")

        true_predictions = []
        true_labels = []

        for prediction, label in zip(predictions, labels):
            true_pred = []
            true_lab = []
            for pred, lab in zip(prediction, label):
                if lab != -100:
                    true_pred.append(pred)
                    true_lab.append(lab)
            true_predictions.append(true_pred)
            true_labels.append(true_lab)

        if not self.id2label:
            raise ValueError("id2label map is None. Expected values.")

        label_list = list(self.id2label.values())
        true_predictions = [
            [label_list[pred] for pred in prediction] for prediction in true_predictions
        ]
        true_labels = [[label_list[lab] for lab in label] for label in true_labels]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)

        if results:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        else:
            raise ValueError("seqeval results dict is empty")
