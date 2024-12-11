from transformers import BertTokenizer, BertModel
import torch


def generate_embedding(text: str) -> list[float]:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    encoding = tokenizer.batch_encode_plus(
        [text],
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


# L12 normalization
def normalize_vector(array):
    norm = torch.norm(torch.tensor(array)).item()
    return [item / norm for item in array]
