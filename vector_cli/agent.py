import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import os

def run_agent(input_file: str, 
              output_folder: str = "outputs",
              text_column: str = "Summary", 
              title_column: str = "book_name", 
              category_column: str = "categories"):
    """
    Create embeddings from dataset based on user-selected columns.
    """

    # Dataset oku
    df = pd.read_csv(input_file)

    required_columns = [text_column, title_column, category_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Boş Summary/Text olanları at
    df = df.dropna(subset=[text_column])

    print(f"Loaded {len(df)} rows.")

    # Embedding modeli yükle
    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = df[text_column].tolist()

    print("Generating embeddings...")
    embeddings = []
    for text in tqdm(texts):
        emb = model.encode(text)
        embeddings.append(emb)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save pickle
    data_to_save = {
        "texts": texts,
        "embeddings": embeddings,
        "titles": df[title_column].tolist(),
        "categories": df[category_column].tolist(),
        "columns": {
            "text_column": text_column,
            "title_column": title_column,
            "category_column": category_column
        }
    }

    with open(os.path.join(output_folder, "vector_data.pkl"), "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Saved {len(embeddings)} embeddings to {output_folder}/vector_data.pkl")
