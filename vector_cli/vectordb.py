from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
import csv

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import pickle
import os

def upload_to_qdrant(pickle_file: str, collection_name: str = "book_summaries"):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    texts = data["texts"]
    embeddings = data["embeddings"]
    titles = data["titles"]
    categories = data["categories"]
    columns = data.get("columns", {
        "text_column": "Summary",
        "title_column": "book_name",
        "category_column": "categories"
    })

    client = QdrantClient(host="localhost", port=6333, timeout=60.0)

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
    )

    print(f"Uploading {len(embeddings)} vectors to Qdrant...")

    batch_size = 100
    points = []

    for idx, (vector, text, title, category) in enumerate(zip(embeddings, texts, titles, categories)):
        points.append(
            PointStruct(
                id=idx,
                vector=vector,
                payload={
                    columns["text_column"]: text,
                    columns["title_column"]: title,
                    columns["category_column"]: category
                }
            )
        )

        if len(points) >= batch_size:
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            points = []

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )

    print(f"Finished uploading {len(embeddings)} vectors to '{collection_name}' collection.")


def inspect_qdrant(pickle_file: str = "outputs/vector_data.pkl",
                   collection_name: str = "book_summaries",
                   limit: int = 5):
    # Pickle dosyasını oku
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    columns = data.get("columns", {
        "text_column": "summary",
        "title_column": "book_name",
        "category_column": "categories"
    })

    title_column = columns["title_column"]
    category_column = columns["category_column"]
    text_column = columns["text_column"]

    client = QdrantClient(host="localhost", port=6333, timeout=60.0)

    collections = client.get_collections().collections
    available_collections = [c.name for c in collections]

    if collection_name not in available_collections:
        print(f"Collection '{collection_name}' does not exist. Available collections: {available_collections}")
        return

    print(f"Inspecting collection '{collection_name}':")

    search_result = client.scroll(
        collection_name=collection_name,
        limit=limit
    )

    points = search_result[0]

    for point in points:
        payload = point.payload
        print("-" * 40)
        print(f"Book Name: {payload.get(title_column, 'N/A')}")
        print(f"Categories: {payload.get(category_column, 'N/A')}")
        print(f"Summary: {payload.get(text_column, 'N/A')[:300]}...")



def search_in_qdrant(query_text: str, 
                     collection_name: str = "book_summaries", 
                     limit: int = 5, 
                     unique: bool = False,
                     pickle_file: str = "outputs/vector_data.pkl"):
    client = QdrantClient(host="localhost", port=6333, timeout=60.0)

    # Pickle dosyasından kolon bilgilerini oku
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    columns = data.get("columns", {
        "text_column": "summary",
        "title_column": "book_name",
        "category_column": "categories"
    })

    title_column = columns["title_column"]
    category_column = columns["category_column"]
    text_column = columns["text_column"]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode(query_text)

    if unique:
        fetch_limit = limit * 10
    else:
        fetch_limit = limit

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=fetch_limit
    )

    seen_books = set()
    results = []

    for hit in search_result:
        payload = hit.payload
        book_name = payload.get(title_column, 'N/A')

        if unique:
            if book_name in seen_books:
                continue
            seen_books.add(book_name)

        results.append({
            "score": hit.score,
            "book_name": book_name,
            "categories": payload.get(category_column, 'N/A'),
            "summary": payload.get(text_column, 'N/A')
        })

        if len(results) >= limit:
            break

    for result in results:
        print("-" * 40)
        print(f"[{result['score']:.4f}] Book Name: {result['book_name']}")
        print(f"Categories: {result['categories']}")
        print(f"Summary: {result['summary'][:300]}...")


def export_qdrant(collection_name: str = "book_summaries", 
                  format: str = "json", 
                  limit: int = 100, 
                  output_path: str = "exported_data",
                  pickle_file: str = "outputs/vector_data.pkl"):
    client = QdrantClient(host="localhost", port=6333, timeout=60.0)

    # Pickle dosyasından kolon bilgilerini oku
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    columns = data.get("columns", {
        "text_column": "summary",
        "title_column": "book_name",
        "category_column": "categories"
    })

    text_column = columns["text_column"]
    title_column = columns["title_column"]
    category_column = columns["category_column"]

    search_result = client.scroll(
        collection_name=collection_name,
        limit=limit
    )

    points = search_result[0]
    payloads = [point.payload for point in points]

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Sıralı export için
    ordered_payloads = []
    for payload in payloads:
        ordered_payload = {
            title_column: payload.get(title_column, ""),
            category_column: payload.get(category_column, ""),
            text_column: payload.get(text_column, "")
        }
        ordered_payloads.append(ordered_payload)

    if format == "json":
        with open(os.path.join(output_path, f"{collection_name}.json"), "w", encoding="utf-8") as f:
            json.dump(ordered_payloads, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(payloads)} records to {output_path}/{collection_name}.json")
    elif format == "csv":
        keys = [title_column, category_column, text_column]
        with open(os.path.join(output_path, f"{collection_name}.csv"), "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(ordered_payloads)
        print(f"Exported {len(payloads)} records to {output_path}/{collection_name}.csv")
    else:
        print("Unsupported format. Use 'json' or 'csv'.")


def clear_qdrant(collection_name: str = "book_summaries"):
    client = QdrantClient(host="localhost", port=6333, timeout=60.0)

    collections = client.get_collections().collections
    available_collections = [c.name for c in collections]

    if collection_name not in available_collections:
        print(f"Collection '{collection_name}' does not exist. Nothing to clear.")
        return

    client.delete_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' has been deleted successfully.")

def search_and_export_in_qdrant(query_text: str, 
                                collection_name: str = "book_summaries", 
                                limit: int = 5, 
                                unique: bool = False,
                                output_path: str = "exported_search",
                                format: str = "json",
                                pickle_file: str = "outputs/vector_data.pkl"):
    client = QdrantClient(host="localhost", port=6333, timeout=60.0)

    # Pickle dosyasından kolon bilgilerini oku
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    columns = data.get("columns", {
        "text_column": "summary",
        "title_column": "book_name",
        "category_column": "categories"
    })

    text_column = columns["text_column"]
    title_column = columns["title_column"]
    category_column = columns["category_column"]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode(query_text)

    if unique:
        fetch_limit = limit * 10
    else:
        fetch_limit = limit

    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=fetch_limit
    )

    seen_books = set()
    results = []

    for hit in search_result:
        payload = hit.payload
        book_name = payload.get(title_column, 'N/A')

        if unique:
            if book_name in seen_books:
                continue
            seen_books.add(book_name)

        results.append({
            title_column: book_name,
            category_column: payload.get(category_column, 'N/A'),
            text_column: payload.get(text_column, 'N/A')
        })

        if len(results) >= limit:
            break

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = f"{collection_name}_search_export.{format}"

    if format == "json":
        with open(os.path.join(output_path, filename), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(results)} search results to {output_path}/{filename}")
    elif format == "csv":
        keys = [title_column, category_column, text_column]
        with open(os.path.join(output_path, filename), "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Exported {len(results)} search results to {output_path}/{filename}")
    else:
        print("Unsupported export format. Use 'json' or 'csv'.")

