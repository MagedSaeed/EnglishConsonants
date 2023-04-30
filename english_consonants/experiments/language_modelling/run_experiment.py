import sys
import click

import datasets

if "." not in sys.path:
    sys.path.append(".")


from english_consonants.processing import process_english, strip_vowels
from english_consonants.experiments.language_modelling.src import constants
from english_consonants.experiments.language_modelling.src.training_pipeline import (
    training_pipeline,
)


@click.command()
@click.option(
    "--vocab_coverage",
    default=constants.DEFAULT_VOCAB_COVERAGE,
    help="Vocab coverage to consider, the tokenizer will consider vocabs that covers this percentage of the running text",
)
@click.option(
    "--sequence_length",
    help="sequence length to consider when tokenizing dataset samples",
    type=int,
    default=None,
)
@click.option(
    "--gpu_devices",
    help="GPU devices indexes to consider if the machine has GPUs. Expected input to be index,index...",
    type=str,
    default=str(constants.GPU_DEVICES),
)
@click.option(
    "--cpu_devices",
    help="CPU devices (processes) to consider if the code will run on CPU. Do not forget to set DEVICE='cpu' in constants for this to work",
    type=int,
    default=constants.CPU_DEVICES,
)
@click.option(
    "--batch_size",
    help="Batch size to consider in various data setups",
    type=int,
    default=constants.DEFAULT_BATCH_SIZE,
)
@click.option(
    "--seqlen_percentile",
    help="Sequence Length Percentile. That is, you would be sure that this percentile of your samples are completely covered",
    type=float,
    default=constants.SEQUENCE_LENGTH_PERCENTILE,
)
def run(
    vocab_coverage,
    gpu_devices,
    cpu_devices,
    batch_size,
    sequence_length=None,
    tokenizer_class=constants.DEFAULT_TOKENIZER_CLASS,
    seqlen_percentile=constants.SEQUENCE_LENGTH_PERCENTILE,
):
    sequence_length_percentile = seqlen_percentile
    gpu_devices = list(map(int, gpu_devices.split(",")))

    dataset_name = "wikitext"
    dataset = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")

    def prepare_example(example):
        example["processed_text"] = process_english(example["text"])
        example["consonants"] = strip_vowels(example["processed_text"])
        return example

    dataset["train"] = (
        dataset["train"]
        .filter(lambda example: len(example["text"].split()) > 20)
        .map(prepare_example)
    )
    dataset["validation"] = (
        dataset["validation"]
        .filter(lambda example: len(example["text"].split()) > 20)
        .map(prepare_example)
    )
    dataset["test"] = (
        dataset["test"]
        .filter(lambda example: len(example["text"].split()) > 20)
        .map(prepare_example)
    )

    train_dataset = list(dataset["train"]["processed_text"])
    val_dataset = list(dataset["validation"]["processed_text"])
    test_dataset = list(dataset["test"]["processed_text"])

    print(
        f"""
        Some of the Dataset Samples before training:
        {constants.NEW_LINE.join(train_dataset[:5])}
        """,
    )

    dataset_name = f"all-{dataset_name}-characters".upper()

    training_pipeline(
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        batch_size=batch_size,
        gpu_devices=gpu_devices,
        cpu_devices=cpu_devices,
        dataset_name=dataset_name,
        vocab_coverage=vocab_coverage,
        tokenizer_class=tokenizer_class,
        sequence_length=sequence_length,
        sequence_length_percentile=sequence_length_percentile,
    )

    consonants_train_dataset = list(dataset["train"]["consonants"])
    consonants_val_dataset = list(dataset["validation"]["consonants"])
    consonants_test_dataset = list(dataset["test"]["consonants"])

    dataset_name = f"consonants-{dataset_name}-characters"

    print(
        f"""
        Some of the Dataset Samples after undotting:
        {constants.NEW_LINE.join(consonants_train_dataset[:5])}
        """,
    )

    training_pipeline(
        batch_size=batch_size,
        gpu_devices=gpu_devices,
        cpu_devices=cpu_devices,
        dataset_name=dataset_name,
        vocab_coverage=vocab_coverage,
        tokenizer_class=tokenizer_class,
        sequence_length=sequence_length,
        val_dataset=consonants_val_dataset,
        test_dataset=consonants_test_dataset,
        train_dataset=consonants_train_dataset,
        sequence_length_percentile=sequence_length_percentile,
    )


if __name__ == "__main__":
    run()
