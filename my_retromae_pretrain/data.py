import os
import random
from copy import deepcopy
from dataclasses import dataclass

import torch.utils.data.dataset
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import DataCollatorForWholeWordMask

from .utils import tensorize_batch


class DatasetForPretraining(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        if os.path.isdir(data_dir):
            datasets = []
            for file in os.listdir(data_dir):
                print(f"Loading {file}")
                file = os.path.join(data_dir, file)
                datasets.append(self.load_dataset(file))
            self.dataset = concatenate_datasets(datasets)
        elif "/" in data_dir:
            self.dataset = load_dataset(data_dir, split='train')
        else:
            print(f"Loading {data_dir}")
            self.dataset = self.load_dataset(data_dir)

    def load_dataset(self, file):
        if file.endswith('.jsonl') or file.endswith('.json'):
            return load_dataset('json', data_files=file)['train']
        elif os.path.isdir(file):
            return Dataset.load_from_disk(file)
        else:
            raise NotImplementedError(f"Not support this file format:{file}")

    def __getitem__(self, item):
        return [self.dataset[item]['question'], self.dataset[item]['answer']]

    def __len__(self):
        return len(self.dataset)


@dataclass
class RetroMAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        for e in examples:

            q_e_trunc = self.tokenizer.encode(e[0], max_length=self.max_seq_length, truncation=True)
            q_tokens = [self.tokenizer._convert_id_to_token(tid) for tid in q_e_trunc]

            c_e_trunc = self.tokenizer.encode(e[1], max_length=self.max_seq_length, truncation=True)
            c_tokens = [self.tokenizer._convert_id_to_token(tid) for tid in c_e_trunc]


            self.mlm_probability = self.decoder_mlm_probability
            mask_set = []
            for _ in range(min(len(c_tokens), 128)):
                mask_set.append(self._whole_word_mask(c_tokens))

            text_matrix_attention_mask = []
            for i in range(len(c_tokens)):
                idx = random.randint(0, min(len(c_tokens), 128) - 1)
                text_decoder_mlm_mask = deepcopy(mask_set[idx])
                text_decoder_mlm_mask[i] = 1
                text_matrix_attention_mask.append(text_decoder_mlm_mask)

            input_ids_batch.append(torch.tensor(q_tokens))
            attention_mask_batch.append(torch.tensor([1] * len(q_e_trunc)))

            c_e_trunc[0] = -100
            c_e_trunc[-1] = -100
            decoder_labels_batch.append(torch.tensor(c_e_trunc))

            decoder_matrix_attention_mask_batch.append(1 - torch.tensor(text_matrix_attention_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        origin_input_ids_batch = input_ids_batch.clone()
        encoder_input_ids_batch = self.torch_mask_tokens(input_ids_batch)
        decoder_labels_batch = tensorize_batch(decoder_labels_batch, -100)
        matrix_attention_mask_batch = tensorize_batch(decoder_matrix_attention_mask_batch, 0)

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "decoder_input_ids": origin_input_ids_batch,
            "decoder_attention_mask": matrix_attention_mask_batch,  # [B,L,L]
            "decoder_labels": decoder_labels_batch,
        }

        return batch
