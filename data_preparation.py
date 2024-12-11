import csv
import json

trec_covid_qa_file_path = "datasets/trec-covid/qrels/test.tsv"
trec_covid_queries_file_path = "datasets/trec-covid/queries.jsonl"

queries = {}
query_to_corpus = {}

# Open the TREC-COVID QA file
with open(trec_covid_qa_file_path, "r") as file:
    reader = csv.reader(file, delimiter="\t")

    next(reader)
    for row in reader:
        [query_id, corpus_id, score] = row

        if query_id not in queries:
            queries[query_id] = {}

        queries[query_id]["corpuses"] = (
            queries[query_id].get("corpuses", []).append(corpus_id)
        )

        if query_id not in query_to_corpus:
            query_to_corpus[query_id] = []

        if len(queries.keys()) == 50:
            break

# Open the TREC-COVID queries file
with open(trec_covid_queries_file_path, "r") as file:
    for line in file:
        data = json.loads(line)
        queries[data["_id"]]["query"] = data["text"]

# Open the TREC-COVID corpus file


output_file = "output.jsonl"

with open(output_file, "w") as file:
    for query_id, query in queries.items():
        file.write(json.dumps({"id": query_id, **query}) + "\n")
