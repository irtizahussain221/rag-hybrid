from transformers import BertTokenizer
import torch
from semchunk import chunk as chunk_via_semchunk
import json

MAX_TOKEN_LIMIT = 480


class Chunker:
    def __init__(self, document: str):
        self.document = document

    def count_tokens(self, text):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        encoding = tokenizer.batch_encode_plus(
            [text],
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids: torch.tensor = encoding["input_ids"]

        return len(input_ids.squeeze().numpy())

    def chunk(self):
        token_count = self.count_tokens(self.document)

        if token_count >= MAX_TOKEN_LIMIT:
            print(f"Document is too large ({token_count} tokens). Chunking...")

            text_chunks = chunk_via_semchunk(
                self.document,
                chunk_size=MAX_TOKEN_LIMIT,
                token_counter=lambda text: self.count_tokens(text),
            )

            return text_chunks
        return [self.document]
