from datasets.dataset_dict import DatasetDict
from huggingface_hub import HfApi, metadata_update
from pathlib import Path
from dataclasses import dataclass, field

from loguru import logger
from datasets import Dataset, Features, List, Value
from seqscore import conll


GELATO_DATA: dict[str, dict[str, Path]] = {
    "level1": {
        "train": Path("data/splits/train1.conll"),
        "dev": Path("data/splits/dev1.conll"),
        "test": Path("data/splits/test1.conll"),
    },
    "level2": {
        "train": Path("data/splits/train2.conll"),
        "dev": Path("data/splits/dev2.conll"),
        "test": Path("data/splits/test2.conll"),
    },
}


@dataclass
class HfUpload:
    hf: HfApi
    level: str
    data: dict[str, Path]
    repo_id: str

    ## HuggingFace metadata
    LICENSE: str = "mit"
    TASK_CATEGORIES: list[str] = field(init=False)
    PRETTY_NAME: str = "The GELATO Dataset for Legislative NER"
    GELATO_LANGUAGE: str = "en"

    GELATO_FEATURES = Features(
        {
            "id": Value("int16"),
            "tokens": List(Value("string")),
            "labels": List(Value("int16")),
        }
    )

    def __post_init__(self):
        self.TASK_CATEGORIES = ["token-classification"]

    def upload(self) -> None:
        data, sequences = self._read_conll_file(Path("data/splits/dev1.conll"))

        dataset_dict: dict[str, Dataset] = {}
        for split, path in self.data.items():
            data, sequences = self._read_conll_file(path)
            dataset_dict[split] = Dataset.from_dict(data)

        DatasetDict(dataset_dict).push_to_hub(  # ty: ignore
            repo_id=self.repo_id,
            config_name=self.level,
            max_shard_size="256mb",
            private=False,
        )
        self._update_gelato_metadata()

    def _read_conll_file(
        self, path: Path
    ) -> tuple[dict[str, list[int] | list[str]], int]:
        mention_encoding = conll.get_encoding("BIO")
        with open(path, "r", encoding="utf-8") as file:
            ingester = conll.CoNLLIngester(mention_encoding)

            data = {"id": [], "tokens": [], "labels": []}

            docs = ingester.ingest(file, str(path), None)
            sequences = 0
            for doc in docs:
                for id, seq in enumerate(doc):
                    sequences += 1
                    data["id"].append(id)
                    data["tokens"].append(list(seq.tokens))
                    data["labels"].append(list(seq.labels))

        return data, sequences

    def _update_gelato_metadata(self) -> None:
        metadata = {
            "language": self.GELATO_LANGUAGE,
            "license": self.LICENSE,
            "task_categories": self.TASK_CATEGORIES,
            "pretty_name": self.PRETTY_NAME,
        }
        metadata_update(repo_id=self.repo_id, metadata=metadata, repo_type="dataset")


if __name__ == "__main__":
    hf = HfApi()

    for subset, data in GELATO_DATA.items():
        HfUpload(hf=hf, level=subset, data=data, repo_id="Wollaston/gelato").upload()
