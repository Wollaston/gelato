# The GELATO Dataset for Legislative NER

This repository contains the code, data, and scores for The Gelato Dataset 
for Legislative NER (LREC 2026).

## CLI

The core of the project is a CLI to make it easy to run experiments on the GELATO dataset.

### Installation

This project uses [uv](https://github.com/astral-sh/uv) to manage the environment and internal dependencies.

With `uv` installed, run `uv sync` in the project root to create a .venv managed
by `uv`. Then, run:

```shell
uv run gelato --help
```

to see commands.

Optionally, install the CLI as a tool on your `$PATH` via:

```shell
uv tool install .
```

and simply run

```shell
gelato --help
```

from anywhere to access the CLI.

### Commands

The CLI has a variety of commands to facilitate working with `gelato`.

For help, run

```shell
uv run gelato --help
```

```
Usage: gelato [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.
  --help                Show this message and exit.

Commands:
  prompt-optimize  Use DSPy to optimize level two type prompts for a level one type
  predict          Load a DSPy-optimized program to predict level two labels 
                   from CoNLL-formatted level one predictions
  fine-tune        Fine-tune a HuggingFace Transformer using `wandb`
  train-model      Train the desired model with the provided parameters
  score            Score a model on the datset at the provided path
  align            Align predictions with tokens if the tokenizer aggregation 
                   pipeline fails. Applies first label wins strategy for
                   aggregation of text and labels. Useful as non-word-based
                   tokenizers sometimes struggle to rebuild and aggregate certain
                   words.
  confusion        Generate confusion matrices from CoNLL-formatted predictions 
                   and their reference counterpart
```

#### prompt-optimize

The `prompt-optimize` command simplifies using `DSPy` to optimize level two type
prompts for each level one type prediction.

```shell
uv run gelato prompt-optimize --help

```

```
Usage: gelato prompt-optimize [OPTIONS] TRAIN_PATH DEV_PATH MODEL

  Use DSPy to optimize level two type prompts for a level one type

Arguments:
  TRAIN_PATH  Path to CoNLL-formatted train dataset  [required]
  DEV_PATH    Path to CoNLL-formatted test dataset  [required]
  MODEL       LLM to prompt as a HuggingFace ID e.g. 'Qwen/Qwen3-32B'
              [required]

Options:
  --level-one-type    [Abstraction|Act|Class|Document|Organization|Person]
                      Level one type to fine-tune a prompt for its
                      level two types  [required]
  --module            [ChainOfThought|Predict]
                      What dspy.Module to use  [required]
  --optimizer         [BetterTogether|BootstrapFewShot|BootstrapFewShotWithRandomSearch|
                      BootstrapFinetune|BootstrapRS|COPRO|Ensemble|InferRules|
                      KNNFewShot|LabeledFewShot|MIPROv2|SIMBA]
                      What dspy.Optimizer [required]
  --window INTEGER    The left-right context window to provide the
                      LLM for each mention  [default: 50]
  --base-url TEXT     URL endpoint for an OpenAI-compatible LLM
                      chat server e.g. 'http://localhost:8000/v1'
                      [default: http://localhost:8000/v1]
  --api-key TEXT      API key for OpenAI LLM endpoint. Defaults to
                      'LOCAL' for self-hosted models that do not
                      require authentication.  [default: LOCAL]
  --k INTEGER         'k' to use when generating kNN if
                      'KNNFewShot' is the Optimizer  [default: 10]
  --help              Show this message and exit.
```

#### predict

Load a DSPy-optimized program to predict level two labels from CoNLL-formatted
level one predictions.

```shell
uv run gelato predict --help
```

```
Usage: gelato predict [OPTIONS] TEST_PATH MODEL

  Load a DSPy-optimized program to predict level two labels from CoNLL-
  formatted level one predictions

Arguments:
  TEST_PATH  Path to CoNLL-formatted test dataset  [required]
  MODEL      LLM to prompt as a HuggingFace ID 
              e.g. 'Qwen/Qwen3-32B' [required]

Options:
  --abstraction-path PATH   Path to optimized Abstraction program  [required]
  --act-path PATH           Path to optimized Act program  [required]
  --class-path PATH         Path to optimized Class program  [required]
  --document-path PATH      Path to optimized Document program  [required]
  --organization-path PATH  Path to optimized Organization program
                            [required]
  --person-path PATH        Path to optimized Person program  [required]
  --output-path PATH        Output path for serialized predictions
                            [required]
  --window INTEGER          The left-right context window to provide the LLM
                            for each mention  [default: 50]
  --base-url TEXT           URL endpoint for an OpenAI-compatible LLM chat
                            server e.g. 'http://localhost:8000/v1'  
                            [default: http://localhost:8000/v1]
  --api-key TEXT            API key for OpenAI LLM endpoint. Defaults to
                            'LOCAL' for self-hosted models that do not require
                            authentication.  [default: LOCAL]
  --help                    Show this message and exit.
```

#### fine-tune

The `fine-tune` command simplifies fine-tuning a HuggingFace Transformer
using `wandb`.

```shell
uv run gelato fine-tune --help
```

```
Usage: gelato fine-tune [OPTIONS] TRAIN_PATH TEST_PATH MODEL

  Fine-tune a HuggingFace Transformer using `wandb`

Arguments:
  TRAIN_PATH  Path to CoNLL-formatted train dataset  [required]
  TEST_PATH   Path to CoNLL-formatted test dataset  [required]
  MODEL       Model to fine-tune as a HuggingFace ID e.g. 'FacebookAI/xlm-
              roberta-base'. Assumes model is compatible with HuggingFace
              transformers.  [required]

Options:
  --output-dir PATH       output directory for wandb logs  [required]
  --wandb-project TEXT    Name of wandb project to track sweeps e.g. 'gelato'
                          [default: gelato]
  --sweeps INTEGER RANGE  Number of wandb sweeps to perform
                          [default: 1; 1<=x<=64]
  --help                  Show this message and exit.
```

#### train-model

Train the desired HuggingFace-compatible transformer model with
the provided parameters

```shell
uv run gelato train-model --help
```

```
Usage: gelato train-model [OPTIONS] MODEL_ID

  Train the desired model with the provided parameters.

Arguments:
  MODEL_ID  The HuggingFace model id of the model to train 
            e.g.'google-bert/bert-base-cased'  [required]

Options:
  --train-path TEXT      The path to the training dataset e.g.
                         'data/train.conll'  [required]
  --dev-path TEXT        The path to the dev dataset e.g. 'data/dev.conll'
                         [required]
  --learning-rate FLOAT  Learning rate of the model e.g. '0.003'  [required]
  --batch-size INTEGER   Learning and eval batch size e.g. '16'  [required]
  --epochs INTEGER       Number of training epochs e.g. '42'  [required]
  --weight-decay FLOAT   Training weight decay e.g. '0.3'  [required]
  --warmup-ratio FLOAT   Training warmup ratio e.g. '0.1'  [required]
  --output-dir TEXT      output directory for wandb logs  [required]
  --help                 Show this message and exit.
```

#### score

Score a model on the datset at the provided path.

```shell
uv run gelato score --help
```

```
Usage: gelato score [OPTIONS] DATASET_PATH MODEL

  Score a model on the datset at the provided path

Arguments:
  DATASET_PATH  Path to CoNLL-formatted dataset to evaluate  [required]
  MODEL         Model to test as a HuggingFace ID e.g.
                'Wollaston/gelato-roberta-large'  [required]

Options:
  --help  Show this message and exit.
```

#### align

Align predictions Applies first label wins strategy for aggregation of
text and labels. Useful as non-word-based tokenizers sometimes struggle
to rebuild and aggregate certain words.

```shell
uv run gelato align --help
```

```
Usage: gelato align [OPTIONS] PREDICTIONS_PATH REFERENCE_PATH

  Align predictions with tokens if the tokenizer aggregation pipeline fails.
  Applies first label wins strategy for aggregation of text and labels. Useful
  as non-word-based tokenizers sometimes struggle to rebuild and aggregate
  certain words.

Arguments:
  PREDICTIONS_PATH  Path to CoNLL-formatted predictions to align  [required]
  REFERENCE_PATH    Path to CoNLL-formatted reference data to align tokens to
                    [required]

Options:
  --help  Show this message and exit.
```

#### confusion

Generate confusion matrices from CoNLL-formatted predictions
and their reference counterpart

```shell
uv run gelato confusion --help
```

```
Usage: gelato confusion [OPTIONS] PREDICTIONS REFERENCES OUTPUT_PATH

  Generate confusion matrices from CoNLL-formatted predictions and their
  reference counterpart

Arguments:
  PREDICTIONS  Path to CoNLL-formatted predictions  [required]
  REFERENCES   Path to CoNLL-formatted references  [required]
  OUTPUT_PATH  Path to save generated confusion matrix  [required]

Options:
  --help  Show this message and exit.
```

## Checkpoints

We released our `gelato` checkpoints on HuggingFace:

  - [bert-base-cased](https://huggingface.co/Wollaston/gelato-bert-base-cased)
  - [bert-large-cased](https://huggingface.co/Wollaston/gelato-bert-large-cased)
  - [roberta-base](https://huggingface.co/Wollaston/gelato-roberta-base)
  - [roberta-large](https://huggingface.co/Wollaston/gelato-roberta-large)
  - [xlm-roberta-base](https://huggingface.co/Wollaston/gelato-xlm-roberta-base)
  - [xlm-roberta-large](https://huggingface.co/Wollaston/gelato-xlm-roberta-large)

## Data

All `gelato` data, including level one and two splits, as well as original annotation data,
can be found in the `data/` folder.

## Optimizers

The final DSPy optimizers can be found in the `optimizers/` folder.

## Scores

The CoNLL-formatted files for our reported scores can be found in the `scores/` folder.
