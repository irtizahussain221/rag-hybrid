import json
from pymilvus import MilvusClient

client = MilvusClient("trec-covid.db")

if client.has_collection(collection_name="trec-covid"):
    client.drop_collection(collection_name="trec-covid")
client.create_collection(
    collection_name="trec-covid",
    dimension=768,
)

path = "../datasets/trec-covid/corpus.jsonl"

with open(path, "r") as file:
    for line in file:
        data = json.loads(line)
        print(data)