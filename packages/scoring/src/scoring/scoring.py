from pathlib import Path

from data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
)


def score(dataset_path: Path, model_id: str) -> None:
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_id,
        add_prefix_space=True,
        stride=0,
        return_overflowing_tokens=True,
        model_max_length=512,
        truncation=True,
        return_offsets_mapping=True,
    )

    dataset = Dataset.from_path(dataset_path).as_hf_dataset()

    model: PreTrainedModel = AutoModelForTokenClassification.from_pretrained(
        model_id, ignore_mismatched_sizes=True
    )

    classifier = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        ignore_labels=[],
        stride=0,
    )

    with open(f"test_eval_{model_id.replace('/', '_')}.conll", "w") as file:
        for seq in dataset:
            pred = classifier(" ".join(seq["tokens"]))
            for p in pred:
                state = "B"
                tokens = p["word"].replace(".", " .").replace(",", " ,").split()
                for tok in tokens:
                    label = (
                        f"{state}-{p['entity_group']}"
                        if p["entity_group"] != "O"
                        else "O"
                    )
                    file.write(f"{tok} {label}\n")
                    state = "I"
            file.write("\n")

    with open(f"test_eval_{model_id.replace('/', '_')}.conll", "r") as file:
        file_content = file.read()

    updated_content = file_content.replace(" O\n##", "")
    with open(f"test_eval_{model_id.replace('/', '_')}.conll", "w") as file:
        file.write(updated_content)
