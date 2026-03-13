from pathlib import Path

from data.data import Dataset


def test_load_open_ner() -> None:
    Dataset.from_path(
        Path(f"{Path.home()}/data/open-ner-1.0/standardized/CONLL02/spa/test.txt")
    )


def test_all_tokens() -> None:
    dataset = Dataset.from_path(
        Path(f"{Path.home()}/data/open-ner-1.0/standardized/CONLL02/spa/test.txt")
    )
    dataset.all_tokens()


def test_mentions() -> None:
    dataset = Dataset.from_path(
        Path(f"{Path.home()}/data/open-ner-1.0/standardized/CONLL02/spa/test.txt")
    )
    print(dataset.all_mentions())
