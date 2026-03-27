from pathlib import Path
from typing import Any

import evaluate
import numpy as np
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


class ModelTrainer(BaseModel):
    model: str
    train_path: str
    dev_path: str
    learning_rate: float
    batch_size: int
    epochs: int
    weight_decay: float
    warmup_ratio: float
    output_dir: str
    id2label: dict[int, str] = {
        0: "O",
        1: "B-Abstraction",
        2: "I-Abstraction",
        3: "B-Act",
        4: "I-Act",
        5: "B-Class",
        6: "I-Class",
        7: "B-Document",
        8: "I-Document",
        9: "B-Organization",
        10: "I-Organization",
        11: "B-Person",
        12: "I-Person",
    }

    label2id: dict[str, int] = {
        "O": 0,
        "B-Abstraction": 1,
        "I-Abstraction": 2,
        "B-Act": 3,
        "I-Act": 4,
        "B-Class": 5,
        "I-Class": 6,
        "B-Document": 7,
        "I-Document": 8,
        "B-Organization": 9,
        "I-Organization": 10,
        "B-Person": 11,
        "I-Person": 12,
    }

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

        train_tokens, train_labels = ingest_conll(Path(self.train_path))
        dev_tokens, dev_labels = ingest_conll(Path(self.dev_path))

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
        dev_dataset = Dataset.from_dict(
            {"tokens": dev_tokens, "ner_tags": dev_labels}, features=features
        )

        return DatasetDict({"train": train_dataset, "dev": dev_dataset})

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

    def train(self) -> None:
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

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=1,
            push_to_hub=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["dev"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.push_to_hub()
        print(trainer.evaluate(tokenized_datasets["dev"]))

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
