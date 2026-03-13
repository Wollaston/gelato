from pathlib import Path
from prompt_tuning.dspy_level2 import Dataset


def test_dataset() -> None:
    Dataset.from_path(Path("tests/test_data/test2.conll"))


def test_context() -> None:
    test = Dataset.from_path(Path("tests/test_data/test2.conll"))
    for doc in test.documents:
        [mention.context for mention in doc.mentions]
