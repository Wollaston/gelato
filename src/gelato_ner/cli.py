from pathlib import Path

import typer
from .prompt_tuning import LEVEL_ONE, MODULE, OPTIMIZER
from typing_extensions import Annotated

app = typer.Typer()


@app.command(help="Use DSPy to optimize level two type prompts for a level one type")
def prompt_optimize(
    train_path: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted train dataset"),
    ],
    dev_path: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted test dataset"),
    ],
    model: Annotated[
        str,
        typer.Argument(help="LLM to prompt as a HuggingFace ID e.g. 'Qwen/Qwen3-32B'"),
    ],
    level_one_type: Annotated[
        LEVEL_ONE,
        typer.Option(
            help="Level one type to fine-tune a prompt for its level two types"
        ),
    ],
    module: Annotated[MODULE, typer.Option(help="What dspy.Module to use")],
    optimizer: Annotated[
        OPTIMIZER,
        typer.Option(help="What dspy.Optimizer to use"),
    ],
    window: Annotated[
        int,
        typer.Option(
            help="The left-right context window to provide the LLM for each mention"
        ),
    ] = 50,
    base_url: Annotated[
        str,
        typer.Option(
            help="URL endpoint for an OpenAI-compatible LLM chat server e.g. 'http://localhost:8000/v1'"
        ),
    ] = "http://localhost:8000/v1",
    api_key: Annotated[
        str,
        typer.Option(
            help="API key for OpenAI LLM endpoint. Defaults to 'LOCAL' for self-hosted models that do not require authentication."
        ),
    ] = "LOCAL",
    k: Annotated[
        int,
        typer.Option(
            help="'k' to use when generating kNN if 'KNNFewShot' is the Optimizer"
        ),
    ] = 10,
):
    from .prompt_tuning import optimize

    optimize(
        train_path=train_path,
        dev_path=dev_path,
        level_one_type=level_one_type,
        window=window,
        module=module,
        optimizer=optimizer,
        model=model,
        api_base=base_url,
        api_key=api_key,
        k=k,
    )


@app.command(
    help="Load a DSPy-optimized program to predict level two labels from CoNLL-formatted level one predictions"
)
def predict(
    test_path: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted test dataset"),
    ],
    model: Annotated[
        str,
        typer.Argument(help="LLM to prompt as a HuggingFace ID e.g. 'Qwen/Qwen3-32B'"),
    ],
    abstraction_path: Annotated[
        Path, typer.Option(help="Path to optimized Abstraction program")
    ],
    act_path: Annotated[Path, typer.Option(help="Path to optimized Act program")],
    class_path: Annotated[Path, typer.Option(help="Path to optimized Class program")],
    document_path: Annotated[
        Path, typer.Option(help="Path to optimized Document program")
    ],
    organization_path: Annotated[
        Path, typer.Option(help="Path to optimized Organization program")
    ],
    person_path: Annotated[Path, typer.Option(help="Path to optimized Person program")],
    output_path: Annotated[
        Path, typer.Option(help="Output path for serialized predictions")
    ],
    window: Annotated[
        int,
        typer.Option(
            help="The left-right context window to provide the LLM for each mention"
        ),
    ] = 50,
    base_url: Annotated[
        str,
        typer.Option(
            help="URL endpoint for an OpenAI-compatible LLM chat server e.g. 'http://localhost:8000/v1'"
        ),
    ] = "http://localhost:8000/v1",
    api_key: Annotated[
        str,
        typer.Option(
            help="API key for OpenAI LLM endpoint. Defaults to 'LOCAL' for self-hosted models that do not require authentication."
        ),
    ] = "LOCAL",
):
    from .prompt_tuning import predict

    predict(
        abstraction_path=abstraction_path,
        act_path=act_path,
        class_path=class_path,
        document_path=document_path,
        organization_path=organization_path,
        person_path=person_path,
        test_path=test_path,
        window=window,
        model=model,
        api_base=base_url,
        api_key=api_key,
        output_path=output_path,
    )


@app.command(help="Fine-tune a HuggingFace Transformer using `wandb`")
def fine_tune(
    train_path: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted train dataset"),
    ],
    test_path: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted test dataset"),
    ],
    model: Annotated[
        str,
        typer.Argument(
            help="Model to fine-tune as a HuggingFace ID e.g. 'FacebookAI/xlm-roberta-base'. Assumes model is compatible with HuggingFace transformers."
        ),
    ],
    output_dir: Annotated[Path, typer.Option(help="output directory for wandb logs")],
    wandb_project: Annotated[
        str,
        typer.Option(help="Name of wandb project to track sweeps e.g. 'gelato'"),
    ] = "cdg-ner",
    sweeps: Annotated[
        int,
        typer.Option(min=1, max=64, help="Number of wandb sweeps to perform"),
    ] = 1,
):
    from .fine_tuning import FineTune

    fine_tuner = FineTune(
        train_path=train_path,
        test_path=test_path,
        model=model,
        wandb_project=wandb_project,
        sweeps=sweeps,
        output_dir=output_dir,
    )

    fine_tuner.sweep()


@app.command(help="Train the desired model with the provided parameters")
def train_model(
    model_id: Annotated[
        str,
        typer.Argument(
            help="The HuggingFace model id of the model to train e.g.'google-bert/bert-base-cased'"
        ),
    ],
    train_path: Annotated[
        str,
        typer.Option(help="The path to the training dataset e.g. 'data/train.conll'"),
    ],
    dev_path: Annotated[
        str, typer.Option(help="The path to the dev dataset e.g. 'data/dev.conll'")
    ],
    learning_rate: Annotated[
        float, typer.Option(help="Learning rate of the model e.g. '0.003'")
    ],
    batch_size: Annotated[
        int, typer.Option(help="Learning and eval batch size e.g. '16'")
    ],
    epochs: Annotated[int, typer.Option(help="Number of training epochs e.g. '42'")],
    weight_decay: Annotated[
        float, typer.Option(help="Training weight decay e.g. '0.3'")
    ],
    warmup_ratio: Annotated[
        float, typer.Option(help="Training warmup ratio e.g. '0.1'")
    ],
    output_dir: Annotated[str, typer.Option(help="output directory for wandb logs")],
):
    from .fine_tuning import ModelTrainer

    ModelTrainer(
        model=model_id,
        train_path=train_path,
        dev_path=dev_path,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        output_dir=output_dir,
    ).train()


@app.command(help="Score a model on the datset at the provided path")
def score(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted dataset to evaluate"),
    ],
    model: Annotated[
        str,
        typer.Argument(
            help="Model to test as a HuggingFace ID e.g. 'huggingface-course/bert-finetuned-ner'"
        ),
    ],
):
    from .scoring import score as score_ner

    score_ner(dataset_path=dataset_path, model_id=model)


@app.command(
    help="Align predictions with tokens if the tokenizer aggregation pipeline fails. Applies first label wins strategy for aggregation of text and labels. Useful as non-word-based tokenizers sometimes struggle to rebuild and aggregate certain words."
)
def align(
    predictions_path: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted predictions to align"),
    ],
    reference_path: Annotated[
        Path,
        typer.Argument(
            help="Path to CoNLL-formatted reference data to align tokens to"
        ),
    ],
):
    from .scoring import align as align_predictions

    with open(predictions_path, "r") as file:
        predictions = file.read()
    with open(reference_path, "r") as file:
        references = file.read()

    updated_preds = align_predictions(predictions, references)
    with open(str(predictions_path.name) + "_aligned.conll", "w") as file:
        file.write(updated_preds)


@app.command(
    help="Generate confusion matrices from CoNLL-formatted predictions and their reference counterpart"
)
def confusion(
    predictions: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted predictions"),
    ],
    references: Annotated[
        Path,
        typer.Argument(help="Path to CoNLL-formatted references"),
    ],
    output_path: Annotated[
        Path,
        typer.Argument(help="Path to save generated confusion matrix"),
    ],
):
    from .fine_tuning import confusion as confusion_matrices

    confusion_matrices(predictions, references, output_path)


def cli() -> None:
    app()
