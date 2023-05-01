from tqdm.auto import tqdm

from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from pytorch_lightning.utilities.model_summary import ModelSummary

from english_consonants.experiments.language_modelling.src import constants
from english_consonants.experiments.language_modelling.src.callbacks import (
    LossMetricsCallback,
)
from english_consonants.experiments.language_modelling.src.models import (
    LitNeuralLanguageModel,
)
from english_consonants.experiments.language_modelling.src.settings import (
    configure_environment,
)

from english_consonants.experiments.language_modelling.src.utils import (
    calculate_perplexity,
    generate_text,
    get_sequence_length,
    get_best_checkpoint,
    get_dataloader,
    get_tokenizer,
    get_vocab_size,
    train_lm,
    get_oovs_rate,
)


def training_pipeline(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    gpu_devices,
    cpu_devices,
    dataset_name,
    vocab_coverage,
    tokenizer_class,
    sequence_length=None,
    dataloader_workers=constants.CPU_COUNT,
    sequence_length_percentile=constants.SEQUENCE_LENGTH_PERCENTILE,
):
    configure_environment()

    print(
        f"""
        Train Samples: {len(train_dataset):,}
        Val Samples: {len(val_dataset):,}
        Test Samples: {len(test_dataset):,}
        """.strip(),
    )

    print(
        f"""
        Calculating vocab size using WordTokenizer:
        """.strip(),
    )

    vocab_size, all_vocab = get_vocab_size(
        dataset=train_dataset,
        vocab_coverage=vocab_coverage,
    )
    # if tokenizer_class != WordTokenizer:
    # add 4 to account for other special chars such as unk and pad.
    # This is severe for char tokenizer but can be okay for others.
    vocab_size += 4
    all_vocab += 4

    print(
        f"""
        Considered Vocab (from WordTokenizer): {vocab_size:,}
        All Vocab (WordTokenizer): {all_vocab:,}
        """.strip(),
    )
    tokenizer = get_tokenizer(
        vocab_size=vocab_size,
        train_dataset=train_dataset,
        tokenizer_class=tokenizer_class,
    )
    print(
        f"""
        Tokenizer Vocab Size: {tokenizer.vocab_size:,}
        """.strip(),
    )
    print(
        f"""
        Calculating Sequence Length:
        """.strip(),
    )
    if sequence_length is None:
        sequence_length = get_sequence_length(
            dataset=list(
                map(
                    tokenizer.split_text,
                    tqdm(train_dataset),
                )
            ),
            percentile=sequence_length_percentile,
        )
    print(
        f"""
        Sequence Length: {sequence_length:,}
        """.strip(),
    )
    print(
        f"""
        Building DataLoaders
        """.strip(),
    )
    train_dataloader = get_dataloader(
        shuffle=True,
        tokenizer=tokenizer,
        dataset=train_dataset,
        batch_size=batch_size,
        workers=dataloader_workers,
        sequence_length=sequence_length,
    )
    val_dataloader = get_dataloader(
        dataset=val_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        workers=dataloader_workers,
        sequence_length=sequence_length,
        drop_last=constants.DEFAULT_BATCH_SIZE < len(val_dataset),
    )
    test_dataloader = get_dataloader(
        tokenizer=tokenizer,
        dataset=test_dataset,
        batch_size=batch_size,
        workers=dataloader_workers,
        sequence_length=sequence_length,
    )
    print(
        f"""
        Train DataLoader: {len(train_dataloader):,}
        Val DataLoader: {len(val_dataloader):,}
        Test DataLoader: {len(test_dataloader):,}
        """.strip(),
    )

    timer_callback = Timer()

    lm_model = LitNeuralLanguageModel(vocab_size=tokenizer.vocab_size)

    print(
        f"""
        {ModelSummary(lm_model)}
        """.strip(),
    )

    wandb_logger = WandbLogger(
        project=f"EnglishNLMs",
        group=dataset_name,
        name=dataset_name + f"_{tokenizer_class.__name__}",
    )
    wandb_logger.watch(lm_model, log="all")
    trainer = train_lm(
        # one_run=True,
        lm_model=lm_model,
        cpu_devices=cpu_devices,
        gpu_devices=gpu_devices,
        dataset_name=dataset_name,
        wandb_logger=wandb_logger,
        callbacks=[timer_callback],
        val_dataloader=val_dataloader,
        vocab_coverage=vocab_coverage,
        max_epochs=constants.MAX_EPOCHS,
        tokenizer_class=tokenizer_class,
        train_dataloader=train_dataloader,
    )
    lm_model = LitNeuralLanguageModel.load_from_checkpoint(
        get_best_checkpoint(
            dataset_name=dataset_name,
            tokenizer_class=tokenizer_class,
        )
    )
    training_perplexity = calculate_perplexity(
        lm_model=lm_model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    perplexity_with_oovs = calculate_perplexity(
        lm_model=lm_model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        sequence_length=sequence_length,
    )

    perplexity_without_oovs = calculate_perplexity(
        ignore_oovs=True,
        lm_model=lm_model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        sequence_length=sequence_length,
    )

    print(
        f"""
        Training Perplexity: {training_perplexity}
        Perplexity with OOVs: {perplexity_with_oovs}
        Perplexity without OOVs: {perplexity_without_oovs:,}
        """.strip(),
    )

    training_oov_rate = get_oovs_rate(dataloader=train_dataloader)
    val_oov_rate = get_oovs_rate(dataloader=train_dataloader)
    test_oov_rate = get_oovs_rate(dataloader=train_dataloader)

    print(
        f"""
        Training OOVs rate: {training_oov_rate}
        Validation OOVs rate: {val_oov_rate}
        Test OOVs rate: {test_oov_rate}
        """.strip(),
    )

    f'{timer_callback.time_elapsed("train"):.2f} seconds'

    print(
        f"""
        Training Time: {f'{timer_callback.time_elapsed("train"):.2f} seconds'}
        """.strip(),
    )

    prompt = "<bos> "

    print(
        generate_text(
            prompt=prompt,
            lm_model=lm_model,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
        ),
    )
