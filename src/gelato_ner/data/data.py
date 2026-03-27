from itertools import chain
from pathlib import Path

from pydantic import BaseModel, computed_field
from seqscore import conll
from datasets import Dataset as HfDataset


class Mention(BaseModel):
    tokens: list[str]
    labels: list[str]
    label: str
    cluster_label: int | None = None

    @computed_field()
    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    def __str__(self):
        return (
            f"{self.text}-{self.label}-{self.cluster_label}"
            if self.cluster_label
            else f"{self.text}-{self.label}"
        )


class Document(BaseModel):
    tokens: list[str]
    labels: list[str]
    mentions: list[Mention]

    def all_tokens(self) -> list[list[str]]:
        all_tokens: list[list[str]] = []
        [all_tokens.append(mention.tokens) for mention in self.mentions]
        return all_tokens

    def all_labels(self) -> list[str]:
        labels: list[str] = []
        [labels.append(mention.label) for mention in self.mentions]
        return labels


class Dataset(BaseModel):
    documents: list[Document]

    @classmethod
    def from_path(cls, path: Path) -> "Dataset":
        mention_encoding = conll.get_encoding("BIO")

        documents: list[Document] = []
        with open(path) as file:
            ingester = conll.CoNLLIngester(mention_encoding)

            docs = ingester.ingest(file, str(path), None)
            for doc in docs:
                for seq in doc:
                    mentions: list[Mention] = []
                    for mention in seq.mentions:
                        span = mention.span
                        tokens = list(seq.tokens[span.start : span.end])
                        labels = list(seq.labels[span.start : span.end])
                        tag = labels[0][2:]
                        mentions.append(
                            Mention(tokens=tokens, labels=labels, label=tag)
                        )
                    documents.append(
                        Document(
                            tokens=list(seq.tokens),
                            labels=list(seq.labels),
                            mentions=mentions,
                        )
                    )
        return cls(documents=documents)

    def all_tokens(self) -> list[list[str]]:
        all_tokens: list[list[str]] = []
        [all_tokens.extend(doc.all_tokens()) for doc in self.documents]
        return all_tokens

    def all_labels(self) -> list[str]:
        all_labels: list[str] = []
        [all_labels.extend(doc.all_labels()) for doc in self.documents]
        return all_labels

    def all_mentions(self) -> list[Mention]:
        all_mentions: list[Mention] = []
        [all_mentions.extend(doc.mentions) for doc in self.documents]
        return all_mentions

    def load_conll(self) -> tuple[list[Mention], list[str]]:
        """
        Reads a BIO-tagged CoNLL documents and returns:

        (1) the list of Mentions decoded from the labels,
        (2) the flat token list.

        Ignores blank lines.
        """
        mentions: list[Mention] = self.all_mentions()
        tokens: list[str] = list(chain.from_iterable(self.all_tokens()))
        return mentions, tokens

    def as_hf_dataset(self) -> HfDataset:
        tokens = [doc.tokens for doc in self.documents]
        labels = [doc.labels for doc in self.documents]
        return HfDataset.from_dict({"tokens": tokens, "labels": labels})
