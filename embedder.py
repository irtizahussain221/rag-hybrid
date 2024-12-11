from transformers import BertTokenizer, BertModel
import torch
from numpy import mean

VECTOR_DIMENSIONS = 768

class Embedder:
    def __init__(self, text: str):
        self.text = text

    def normalize_vector(self, vector: list[float]) -> list[float]:
        norm = torch.norm(torch.tensor(vector)).item()
        return [item / norm for item in vector]

    def generate_embedding(self) -> list[float]:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        encoding = tokenizer.batch_encode_plus(
            [self.text],
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            word_embeddings = (
                outputs.last_hidden_state
            )  # (batch_size, sequence_length, hidden)
            sentence_embedding = (
                word_embeddings.mean(dim=1).flatten().tolist()
            )  # Mean pooling

        return sentence_embedding

    def embed(self):
        return self.normalize_vector(self.generate_embedding())


class ChunkEmbedder:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        
    def embed(self) -> list[float]:
        vectors = [Embedder(chunk).embed() for chunk in self.chunks]
        return mean(vectors, axis=0).tolist()