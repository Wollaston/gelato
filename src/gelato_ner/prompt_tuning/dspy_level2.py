from pathlib import Path
from typing import Any, Literal, Optional, cast
from uuid import uuid4

import dspy
from dspy.teleprompt import Teleprompter
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from seqscore import conll

MODULE = Literal["ChainOfThought", "Predict"]

OPTIMIZER = Literal[
    "BetterTogether",
    "BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch",
    "BootstrapFinetune",
    "BootstrapRS",
    "COPRO",
    "Ensemble",
    "InferRules",
    "KNNFewShot",
    "LabeledFewShot",
    "MIPROv2",
    "SIMBA",
]

LEVEL_ONE = Literal["Abstraction", "Act", "Class", "Document", "Organization", "Person"]
LEVEL_TWO = Literal[
    "Agency",
    "Amendment",
    "Association",
    "Bill",
    "Code",
    "Committee",
    "Doctrine",
    "Fund",
    "Infrastructure",
    "InternationalInstitution",
    "LegislativeBody",
    "Locality",
    "Member",
    "Misc",
    "Name",
    "Nation",
    "Non-ProtectedClass",
    "Parenthetical",
    "Program",
    "ProtectedClass",
    "PublicAct",
    "Reference",
    "Report",
    "Session",
    "Specification",
    "State",
    "System",
    "Title",
]

LEVEL2to1_MAP: dict[str, str] = {
    "Doctrine": "Abstraction",
    "Fund": "Abstraction",
    "Infrastructure": "Abstraction",
    "Misc": "Abstraction",
    "Program": "Abstraction",
    "Session": "Abstraction",
    "Specification": "Abstraction",
    "System": "Abstraction",
    "Amendment": "Act",
    "PublicAct": "Act",
    "Non-ProtectedClass": "Class",
    "ProtectedClass": "Class",
    "Bill": "Document",
    "Code": "Document",
    "Parenthetical": "Document",
    "Reference": "Document",
    "Report": "Document",
    "Agency": "Organization",
    "Association": "Organization",
    "Committee": "Organization",
    "InternationalInstitution": "Organization",
    "LegislativeBody": "Organization",
    "Locality": "Organization",
    "Nation": "Organization",
    "State": "Organization",
    "Member": "Person",
    "Name": "Person",
    "Title": "Person",
}

LEVEL1to2_MAP: dict[str, list[str]] = {
    "Abstraction": [
        "Doctrine",
        "Fund",
        "Infrastructure",
        "Misc",
        "Program",
        "Session",
        "Specification",
        "System",
    ],
    "Act": ["Amendment", "PublicAct"],
    "Class": ["Non-ProtectedClass", "ProtectedClass"],
    "Document": [
        "Bill",
        "Code",
        "Parenthetical",
        "Reference",
        "Report",
    ],
    "Organization": [
        "Agency",
        "Association",
        "Committee",
        "InternationalInstitution",
        "LegislativeBody",
        "Locality",
        "Nation",
        "State",
    ],
    "Person": ["Member", "Name", "Title"],
}


class Mention(BaseModel):
    tokens: list[str]
    labels: list[str]
    context: str
    tag: str
    idx: int
    level_one: Optional[LEVEL_ONE] = None
    level_two: Optional[LEVEL_TWO] = None

    def as_example(self) -> dspy.Example:
        return dspy.Example(
            mention=" ".join(self.tokens),
            context=self.context,
            level_two=self.level_two,
            possible_tags=LEVEL1to2_MAP[self.level_one] if self.level_one else "",
        ).with_inputs("mention", "context", "possible_tags")


class Document(BaseModel):
    tokens: list[str]
    mentions: list[Mention]

    def filtered_mentions(self, level_one: LEVEL_ONE):
        return list(
            filter(lambda mention: mention.level_one == level_one, self.mentions)
        )


class Dataset(BaseModel):
    documents: list[Document]

    @classmethod
    def from_path(cls, path: Path, window: int = 50) -> "Dataset":
        logger.info(f"Loading Dataset from Path: {path}")
        mention_encoding = conll.get_encoding("BIO")

        idx = 0

        documents: list[Document] = []
        with open(path) as file:
            ingester = conll.CoNLLIngester(mention_encoding)

            docs = ingester.ingest(file, str(path), None)
            for doc in docs:
                for seq in doc:
                    length = len(seq.tokens)
                    mentions: list[Mention] = []
                    for mention in seq.mentions:
                        span = mention.span
                        tokens = list(seq.tokens[span.start : span.end])
                        labels = list(seq.labels[span.start : span.end])

                        context_start = (
                            span.start - window if span.start - window > 0 else 0
                        )
                        context_end = (
                            span.end + window if span.end + window < length else length
                        )
                        context = " ".join(list(seq.tokens[context_start:context_end]))

                        tag: str = labels[0][2:]
                        level_one, level_two = tag.split("_")
                        level_one = cast(LEVEL_ONE, level_one)
                        level_two = cast(LEVEL_TWO, level_two)

                        if LEVEL2to1_MAP[level_two] != level_one:
                            raise ValueError(
                                f"Could not validate LEVEL_TWO tag {level_two} with LEVEL_ONE {level_one}.\nValid tag map: {LEVEL2to1_MAP}"
                            )

                        mentions.append(
                            Mention(
                                tokens=tokens,
                                labels=labels,
                                context=context,
                                tag=tag,
                                idx=idx,
                                level_one=level_one,
                                level_two=level_two,
                            )
                        )
                        idx += 1
                    documents.append(
                        Document(tokens=list(seq.tokens), mentions=mentions)
                    )
        return cls(documents=documents)

    @classmethod
    def from_predictions(cls, path: Path, window: int = 50) -> "Dataset":
        logger.info(f"Loading Predictions from Path: {path}")
        mention_encoding = conll.get_encoding("BIO")

        idx = 0

        documents: list[Document] = []
        with open(path) as file:
            ingester = conll.CoNLLIngester(mention_encoding)

            docs = ingester.ingest(file, str(path), None)
            for doc in docs:
                for seq in doc:
                    length = len(seq.tokens)
                    mentions: list[Mention] = []
                    for mention in seq.mentions:
                        span = mention.span
                        tokens = list(seq.tokens[span.start : span.end])
                        labels = list(seq.labels[span.start : span.end])

                        context_start = (
                            span.start - window if span.start - window > 0 else 0
                        )
                        context_end = (
                            span.end + window if span.end + window < length else length
                        )
                        context = " ".join(list(seq.tokens[context_start:context_end]))

                        tag: str = labels[0][2:]
                        level_one = cast(LEVEL_ONE, tag)

                        mentions.append(
                            Mention(
                                tokens=tokens,
                                labels=labels,
                                context=context,
                                tag=tag,
                                idx=idx,
                                level_one=level_one,
                            )
                        )
                        idx += 1
                    documents.append(
                        Document(tokens=list(seq.tokens), mentions=mentions)
                    )
        return cls(documents=documents)

    def as_examples(self, level_one: Optional[LEVEL_ONE] = None) -> list[dspy.Example]:
        examples: list[dspy.Example] = []
        if level_one:
            mentions: list[Mention] = []
            [
                mentions.extend(doc.filtered_mentions(level_one=level_one))
                for doc in self.documents
            ]
            [examples.append(mention.as_example()) for mention in mentions]
        else:
            for doc in self.documents:
                [examples.append(mention.as_example()) for mention in doc.mentions]

        return examples

    def mentions(self, level_one: Optional[LEVEL_ONE] = None) -> list[Mention]:
        mentions: list[Mention] = []
        if level_one:
            [
                mentions.extend(doc.filtered_mentions(level_one=level_one))
                for doc in self.documents
            ]
        else:
            for doc in self.documents:
                [mentions.append(mention) for mention in doc.mentions]

        return mentions


class GelatoSignature(dspy.Signature):
    """
    Extract contiguous tokens referring to members of congress, titles, or simple names, if any,
    from a list of string tokens. Output a list of tokens.
    """

    mention: str = dspy.InputField(desc="the person mention to extract the type from")
    context: str = dspy.InputField(desc="the context surrounding the mention")
    possible_tags: list[str] = dspy.InputField(desc="List of possible level 2 tags")
    tag: str = dspy.OutputField(
        desc="the type of person mention. MUST BE ONE OF THE POSSIBLE TAGS PROVDED."
    )


class Optimizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    train_set: list[dspy.Example]
    dev_set: list[dspy.Example]
    level_one: LEVEL_ONE
    window: int
    module: MODULE
    optimizer: OPTIMIZER
    model: str
    api_base: str
    api_key: str
    k: int
    uuid: str = Field(default_factory=lambda: str(uuid4()))

    def configure_lm(self) -> dspy.LM:
        return dspy.LM(
            model=f"openai/{self.model}",
            api_base=self.api_base,
            api_key=self.api_key,
            model_type="chat",
            num_retries=16,
            timeout=3000,
        )

    def configure_module(self) -> dspy.Module:
        logger.debug(f"Configuring dspy.Module {self.module}")
        match self.module:
            case "ChainOfThought":
                return dspy.ChainOfThought(GelatoSignature)
            case "Predict":
                return dspy.Predict(GelatoSignature)

    def configure_optimizer(self) -> Teleprompter:
        logger.debug(f"Configuring dspy.Optimizer {self.optimizer}")
        extraction_correctness_metric = self.extraction_correctness_metric
        match self.optimizer:
            case "BetterTogether":
                return dspy.BetterTogether(
                    metric=extraction_correctness_metric,
                )
            case "BootstrapFewShot":
                return dspy.BootstrapFewShot(
                    metric=extraction_correctness_metric,
                )
            case "BootstrapFewShotWithRandomSearch":
                return dspy.BootstrapFewShotWithRandomSearch(
                    metric=extraction_correctness_metric,
                )
            case "BootstrapFinetune":
                return dspy.BootstrapFinetune(
                    metric=extraction_correctness_metric,
                )
            case "BootstrapRS":
                return dspy.BootstrapRS(
                    metric=extraction_correctness_metric,
                )
            case "COPRO":
                return dspy.COPRO(
                    metric=extraction_correctness_metric,
                    auto="medium",
                )
            case "Ensemble":
                return dspy.Ensemble()
            case "InferRules":
                return dspy.InferRules(
                    metric=extraction_correctness_metric,
                    auto="medium",
                )
            case "KNNFewShot":
                from sentence_transformers import SentenceTransformer

                return dspy.KNNFewShot(
                    k=self.k,
                    trainset=self.train_set,
                    vectorizer=dspy.Embedder(
                        SentenceTransformer("all-MiniLM-L6-v2").encode
                    ),
                )
            case "LabeledFewShot":
                return dspy.LabeledFewShot()
            case "MIPROv2":
                return dspy.MIPROv2(
                    metric=extraction_correctness_metric,
                    auto="medium",
                )
            case "SIMBA":
                return dspy.SIMBA(
                    metric=self.simba_correctness_metric,
                )

    def extraction_correctness_metric(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace=None,
    ) -> bool:
        """
        Computes correctness of entity extraction predictions.

        Args:
            example (dspy.Example): The dataset example containing expected people entities.
            prediction (dspy.Prediction): The prediction from the DSPy people extraction program.
            trace: Optional trace object for debugging.

        Returns:
            bool: True if predictions match expectations, False otherwise.
        """
        return prediction.tag == example.level_two

    def simba_correctness_metric(
        self,
        example: dspy.Example,
        predictions: dict[str, Any],
        trace=None,
    ) -> float:
        """
        Computes correctness of entity extraction predictions.

        Args:
            example (dspy.Example): The dataset example containing expected people entities.
            prediction (dspy.Prediction): The prediction from the DSPy people extraction program.
            trace: Optional trace object for debugging.

        Returns:
            bool: True if predictions match expectations, False otherwise.
        """
        score: float = 0.0
        for pred in predictions.values():
            if pred == example.level_two:
                score += 1.0
        return score / len(predictions)

    def optimize(self) -> None:
        logger.info(f"Configuring DSPy Entity Extraction for {self.level_one}")
        logger.debug(f"Train set length: {len(self.train_set)}")
        logger.debug(f"Dev set length: {len(self.dev_set)}")

        dspy.configure(lm=self.configure_lm())
        module = self.configure_module()
        optimizer = self.configure_optimizer()
        optimizer = optimizer.compile(module, trainset=self.train_set)

        evaluate_correctness = dspy.Evaluate(
            devset=self.dev_set,
            metric=self.extraction_correctness_metric
            if self.optimizer != "SIMBA"
            else self.simba_correctness_metric,
            num_threads=24,
            display_progress=True,
            display_table=True,
            save_as_csv=f"dev_{self.level_one}_{self.uuid}.csv",
        )

        logger.info("Optimizing")
        result = evaluate_correctness(module)

        logger.info(f"Result for {self.level_one}: {result}")
        with open(f"{self.level_one}_{self.uuid}.txt", "w") as file:
            logger.info("Saving result")
            file.write(f"Score: {result.score}\n")
            file.write(f"level_one: {self.level_one}\n")
            file.write(f"window: {self.window}\n")
            file.write(f"module: {self.module}\n")
            file.write(f"optimizer: {self.optimizer}\n")
            file.write(f"model: {self.model}\n")
            file.write(f"api_base: {self.api_base}\n")
            file.write(f"k: {self.k}\n")
            file.write(f"uuid: {self.uuid}\n")
            file.write("\nRESULTS\n")
            for res in result.results:
                file.write(f"{res[0]['mention']}\t{res[1]['tag']}\t{res[2]}\n")

        logger.info(f"Saving optimized model with uuid {self.uuid}")
        module.save(
            f"optimized_{self.level_one}_{self.module}_{self.optimizer}_{self.window}_{self.uuid}",
            save_program=True,
        )


def optimize(
    train_path: Path,
    dev_path: Path,
    level_one_type: LEVEL_ONE,
    window: int,
    module: MODULE,
    optimizer: OPTIMIZER,
    model: str,
    api_base: str,
    api_key: str,
    k: int,
):
    logger.info(
        f"Optimizing prompt for level one type {level_one_type} with Module {module} and Optimizer {optimizer} with window {window}"
    )
    train_set = Dataset.from_path(Path(train_path), window=window)
    dev_set = Dataset.from_path(Path(dev_path), window=window)

    prompt_optimizer = Optimizer(
        train_set=train_set.as_examples(level_one=level_one_type),
        dev_set=dev_set.as_examples(level_one=level_one_type),
        level_one=level_one_type,
        window=window,
        module=module,
        optimizer=optimizer,
        model=model,
        api_base=api_base,
        api_key=api_key,
        k=k,
    )

    prompt_optimizer.optimize()


def predict(
    abstraction_path: Path,
    act_path: Path,
    class_path: Path,
    document_path: Path,
    organization_path: Path,
    person_path: Path,
    test_path: Path,
    window: int,
    model: str,
    api_base: str,
    api_key: str,
    output_path: Path,
):
    logger.info(f"Evaluating {test_path} with window of {window}")

    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )

    data = Dataset.from_predictions(test_path, window=window)

    dspy.configure(
        lm=dspy.LM(
            model=f"openai/{model}",
            api_base=api_base,
            api_key=api_key,
            model_type="chat",
            num_retries=16,
            timeout=3000,
        )
    )

    abstraction_program = dspy.load(str(abstraction_path))
    act_program = dspy.load(str(act_path))
    class_program = dspy.load(str(class_path))
    document_program = dspy.load(str(document_path))
    organization_program = dspy.load(str(organization_path))
    person_program = dspy.load(str(person_path))

    def mention_ids(mentions: list[Mention]) -> list[int]:
        return [mention.idx for mention in mentions]

    abstraction_mentions = data.mentions(level_one="Abstraction")
    abstraction_ids = mention_ids(abstraction_mentions)
    abstraction_predictions = abstraction_program.batch(
        examples=data.as_examples(level_one="Abstraction"),
        num_threads=8,
        return_failed_examples=True,
    )

    act_mentions = data.mentions(level_one="Act")
    act_ids = mention_ids(act_mentions)
    act_predictions = act_program.batch(
        examples=data.as_examples(level_one="Act"),
        num_threads=8,
        return_failed_examples=True,
    )

    class_mentions = data.mentions(level_one="Class")
    class_ids = mention_ids(class_mentions)
    class_predictions = class_program.batch(
        examples=data.as_examples(level_one="Class"),
        num_threads=8,
        return_failed_examples=True,
    )

    document_mentions = data.mentions(level_one="Document")
    document_ids = mention_ids(document_mentions)
    document_predictions = document_program.batch(
        examples=data.as_examples(level_one="Document"),
        num_threads=8,
        return_failed_examples=True,
    )

    organization_mentions = data.mentions(level_one="Organization")
    organization_ids = mention_ids(organization_mentions)
    organization_predictions = organization_program.batch(
        examples=data.as_examples(level_one="Organization"),
        num_threads=8,
        return_failed_examples=True,
    )

    person_mentions = data.mentions(level_one="Person")
    person_ids = mention_ids(person_mentions)
    person_predictions = person_program.batch(
        examples=data.as_examples(level_one="Person"),
        num_threads=8,
        return_failed_examples=True,
    )

    predictions: list[tuple[int, dspy.Example]] = []
    for pred, idx in zip(abstraction_predictions[0], abstraction_ids):
        predictions.append((idx, pred))
    for pred, idx in zip(act_predictions[0], act_ids):
        predictions.append((idx, pred))
    for pred, idx in zip(class_predictions[0], class_ids):
        predictions.append((idx, pred))
    for pred, idx in zip(document_predictions[0], document_ids):
        predictions.append((idx, pred))
    for pred, idx in zip(organization_predictions[0], organization_ids):
        predictions.append((idx, pred))
    for pred, idx in zip(person_predictions[0], person_ids):
        predictions.append((idx, pred))

    predictions = sorted(predictions, key=lambda x: x[0])
    for pred, mention in zip(predictions, data.mentions()):
        mention.tag += "_" + pred[1].tag

        def update_tag(label: str, tag: str) -> str:
            return label + "_" + tag

        mention.labels = [update_tag(label, pred[1].tag) for label in mention.labels]
        mention.level_two = pred[1].tag

    idx = 0
    inside = False
    updated_doc: str = ""

    mention_encoding = conll.get_encoding("BIO")
    with open(test_path, "r") as file:
        ingester = conll.CoNLLIngester(mention_encoding)

        mentions = data.mentions()
        docs = ingester.ingest(file, str(test_path), None)

        for doc in docs:
            for seq in doc:
                for pair in seq.tokens_with_labels():
                    pair = list(pair)
                    if pair[1][:2] == "B-" and inside:
                        idx += 1
                        pair[1] += "_" + mentions[idx].level_two
                        inside = True
                    elif pair[1][:2] == "B-":
                        pair[1] += "_" + mentions[idx].level_two
                        inside = True
                    elif pair[1][:2] == "I-":
                        pair[1] += "_" + mentions[idx].level_two
                        inside = True
                    elif inside:
                        inside = False
                        idx += 1
                    updated_doc += f"{pair[0]}\t{pair[1]}\n"
                updated_doc += "\n"

    with open(output_path, "w") as file:
        file.write(updated_doc)
