import sys
import click

import datasets

if "." not in sys.path:
    sys.path.append(".")


from english_consonants.processing import (
    characters_frequency,
    process_english,
    mask_vowels,
    tokens_frequency,
    calculate_entropy,
)
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
    dataset = datasets.load_dataset("wikitext", "wikitext-2-v1")

    def prepare_example(example):
        example["processed_text"] = process_english(example["text"])
        example["consonants"] = mask_vowels(example["processed_text"])
        example["masked_consonants"] = mask_vowels(example["processed_text"], mask="a")
        return example

    dataset["train"] = (
        dataset["train"]
        # .filter(lambda example: len(example["text"].split()) > 10)
        .map(prepare_example)
    )
    dataset["validation"] = (
        dataset["validation"]
        # .filter(lambda example: len(example["text"].split()) > 10)
        .map(prepare_example)
    )
    dataset["test"] = (
        dataset["test"]
        # .filter(lambda example: len(example["text"].split()) > 10)
        .map(prepare_example)
    )

    train_dataset = list(dataset["train"]["processed_text"])
    val_dataset = list(dataset["validation"]["processed_text"])
    test_dataset = list(dataset["test"]["processed_text"])

    train_tokens_frequency = tokens_frequency(dataset=tuple(train_dataset))

    train_characters_frequency = characters_frequency(dataset=tuple(train_dataset))

    print(f"number of train vocabs: {len(train_tokens_frequency):,}")
    print(f"number of train tokens: {sum(train_tokens_frequency.values()):,}")

    print(f"number of train unique characters: {len(train_characters_frequency):,}")
    print(
        f"number of train all characters: {sum(train_characters_frequency.values()):,}"
    )

    train_words_entropy = calculate_entropy(tokens_frequency=train_tokens_frequency)
    train_characters_entropy = calculate_entropy(
        tokens_frequency=train_characters_frequency
    )

    print(f"train words entropy: {train_words_entropy:,}")
    print(f"train chars entropy: {train_characters_entropy:,}")

    print(
        f"""
        Some of the Dataset Samples before training:
        {constants.NEW_LINE.join(train_dataset[:5])}
        """.strip(),
    )

    training_pipeline(
        batch_size=batch_size,
        gpu_devices=gpu_devices,
        cpu_devices=cpu_devices,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        vocab_coverage=vocab_coverage,
        tokenizer_class=tokenizer_class,
        sequence_length=sequence_length,
        dataset_name=f"all-{dataset_name}-chars",
        sequence_length_percentile=sequence_length_percentile,
    )

    print("training on consonants english")

    consonants_train_dataset = list(dataset["train"]["consonants"])
    consonants_val_dataset = list(dataset["validation"]["consonants"])
    consonants_test_dataset = list(dataset["test"]["consonants"])

    consonants_train_tokens_frequency = tokens_frequency(
        dataset=tuple(consonants_train_dataset)
    )

    consonants_train_characters_frequency = characters_frequency(
        dataset=tuple(consonants_train_dataset)
    )

    print(
        f"number of consonants train vocabs: {len(consonants_train_tokens_frequency):,}"
    )
    print(
        f"number of consonants train tokens: {sum(consonants_train_tokens_frequency.values()):,}"
    )

    print(
        f"number of consonants train unique characters: {len(consonants_train_characters_frequency):,}"
    )
    print(
        f"number of consonants train all characters: {sum(consonants_train_characters_frequency.values()):,}"
    )

    consonants_train_words_entropy = calculate_entropy(
        tokens_frequency=consonants_train_tokens_frequency
    )
    consonants_train_characters_entropy = calculate_entropy(
        tokens_frequency=consonants_train_characters_frequency
    )

    print(f"train consonants words entropy: {consonants_train_words_entropy:,}")
    print(f"train consonants chars entropy: {consonants_train_characters_entropy:,}")

    print(
        f"""
        Some of the Dataset Samples after deleting vowels:
        {constants.NEW_LINE.join(consonants_train_dataset[:5])}
        """.strip(),
    )

    training_pipeline(
        batch_size=batch_size,
        gpu_devices=gpu_devices,
        cpu_devices=cpu_devices,
        vocab_coverage=vocab_coverage,
        tokenizer_class=tokenizer_class,
        sequence_length=sequence_length,
        val_dataset=consonants_val_dataset,
        test_dataset=consonants_test_dataset,
        train_dataset=consonants_train_dataset,
        dataset_name=f"consonants-{dataset_name}-chars",
        sequence_length_percentile=sequence_length_percentile,
    )

    print("training on masked consonants english")

    masked_consonants_train_dataset = list(dataset["train"]["masked_consonants"])
    masked_consonants_val_dataset = list(dataset["validation"]["masked_consonants"])
    masked_consonants_test_dataset = list(dataset["test"]["masked_consonants"])

    masked_consonants_train_tokens_frequency = tokens_frequency(
        dataset=tuple(masked_consonants_train_dataset)
    )

    masked_consonants_train_characters_frequency = characters_frequency(
        dataset=tuple(masked_consonants_train_dataset)
    )

    print(
        f"number of masked consonants train vocabs: {len(masked_consonants_train_tokens_frequency):,}"
    )
    print(
        f"number of masked consonants train tokens: {sum(masked_consonants_train_tokens_frequency.values()):,}"
    )

    print(
        f"number of masked consonants train unique characters: {len(masked_consonants_train_characters_frequency):,}"
    )
    print(
        f"number of masked consonants train all characters: {sum(masked_consonants_train_characters_frequency.values()):,}"
    )

    masked_consonants_train_words_entropy = calculate_entropy(
        tokens_frequency=masked_consonants_train_tokens_frequency
    )
    masked_consonants_train_characters_entropy = calculate_entropy(
        tokens_frequency=masked_consonants_train_characters_frequency
    )

    print(
        f"train masked consonants words entropy: {masked_consonants_train_words_entropy:,}"
    )
    print(
        f"train masked consonants chars entropy: {masked_consonants_train_characters_entropy:,}"
    )

    print(
        f"""
        Some of the Dataset Samples after masking:
        {constants.NEW_LINE.join(masked_consonants_train_dataset[:5])}
        """.strip(),
    )

    training_pipeline(
        batch_size=batch_size,
        gpu_devices=gpu_devices,
        cpu_devices=cpu_devices,
        vocab_coverage=vocab_coverage,
        tokenizer_class=tokenizer_class,
        sequence_length=sequence_length,
        val_dataset=masked_consonants_val_dataset,
        test_dataset=masked_consonants_test_dataset,
        train_dataset=masked_consonants_train_dataset,
        sequence_length_percentile=sequence_length_percentile,
        dataset_name=f"masked-consonants-{dataset_name}-chars",
    )


if __name__ == "__main__":
    run()
