import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        sequence_length,
        use_tqdm=True,
    ):
        super().__init__()
        dataset = tqdm(dataset) if use_tqdm else dataset
        self.encoded_dataset = []
        for document in dataset:
            if not document:
                continue
            tokenized_document = tokenizer.tokenize_from_splits(document)
            if len(tokenized_document) < (sequence_length - 2):
                tokenized_document += [tokenizer.pad_token] * (
                    sequence_length - 2 - len(tokenized_document)
                )
            encoded_document = [tokenizer.token_to_id("<bos>")]
            for token in tokenized_document[: sequence_length - 2]:
                encoded_document.append(tokenizer.token_to_id(token))
            encoded_document.append(tokenizer.token_to_id("<eos>"))
            self.encoded_dataset.append(encoded_document)

    def __getitem__(self, index):
        inputs = self.encoded_dataset[index][:-1]
        outputs = self.encoded_dataset[index][1:]
        return inputs, outputs

    def __len__(self):
        return len(self.encoded_dataset)


def dataset_collate_fn(items):
    returned_inputs, returned_outputs = [], []
    for inputs, outputs in items:
        returned_inputs.append(inputs)
        returned_outputs.append(outputs)
    returned_inputs = torch.LongTensor(returned_inputs)
    returned_outputs = torch.LongTensor(returned_outputs)
    return returned_inputs, returned_outputs
