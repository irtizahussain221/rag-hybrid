import json
from chunker import Chunker
from embedder import ChunkEmbedder
from vector_store import Vector, VectorStore

trec_covid_corpus = "datasets/trec-covid/corpus.jsonl"
nf_corpus = "datasets/nfcorpus/corpus.jsonl"

trec_covid_vectors = []
nf_corpus_vectors = []
trec_covid_store = VectorStore(db_name="trec_covid.db", collection_name="trec_covid")
nf_corpus_store = VectorStore(db_name="nf_corpus.db", collection_name="nf_corpus")

with open(trec_covid_corpus, "r") as file:
    for line in file:
        data: list[Vector] = json.loads(line)

        chunks = Chunker(data["text"]).chunk()
        vector = ChunkEmbedder(chunks).embed()

        trec_covid_vectors.append(Vector(id=data["id"], vector=vector))

    trec_covid_store.upsert(vectors=trec_covid_vectors)

with open(nf_corpus, "r") as file:
    for line in file:
        data: list[Vector] = json.loads(line)

        chunks = Chunker(data["text"]).chunk()
        vector = ChunkEmbedder(chunks).embed()

        nf_corpus_vectors.append(Vector(id=data["id"], vector=vector))

    nf_corpus_store.upsert(vectors=nf_corpus_vectors)
