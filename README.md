# 📚 Vector CLI - Search, Manage, and Export Your Text Embeddings

**Vector CLI** is a Python-based command-line tool to manage datasets with text embeddings, store them into a vector database (Qdrant), search over them, and export results.

This project is **dynamic**, **dataset-agnostic**, and designed for **production-ready** embedding pipelines.

---

## 🚀 Features

- Process any CSV dataset with customizable columns (text, title, category).
- Generate embeddings using a free open-source model (`all-MiniLM-L6-v2`).
- Store embeddings into a local Qdrant instance (runs via Docker).
- Search by text queries with optional duplicate filtering.
- Export full collections or specific search results into CSV or JSON files.
- Clear (delete) collections easily.
- Fully dynamic column mapping — works with any dataset without code changes.
- CLI-first design, perfect for integration into larger systems.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your_username/vector-cli.git
cd vector-cli
```

### 2. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run Qdrant with Docker

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## ⚙️ Available CLI Commands

> All commands assume you're inside the `vector-cli/` directory.

### Create Embeddings (`agent`)

```bash
python3 vector_cli/cli.py agent path/to/your_dataset.csv --text-column Summary --title-column Book_Name --category-column Categories
```

### Upload Embeddings to Qdrant (`upload`)

```bash
python3 vector_cli/cli.py upload outputs/vector_data.pkl book_summaries
```

### Inspect Collection (`inspect`)

```bash
python3 vector_cli/cli.py inspect book_summaries --limit 5
```

### Search Collection (`search`)

```bash
python3 vector_cli/cli.py search "future dystopia" book_summaries --limit 10 --unique
```

- `--unique`: Optional. Return only one result per unique title.

### Export Full Collection (`export`)

```bash
python3 vector_cli/cli.py export book_summaries --format csv --limit 100
```

### Export Search Results (`search-export`)

```bash
python3 vector_cli/cli.py search-export "mental health" book_summaries --limit 10 --unique --format json
```

### Clear Collection (`clear`)

```bash
python3 vector_cli/cli.py clear book_summaries
```

---

## 📦 Project Structure

```plaintext
vector-cli/
├── vector_cli/
│   ├── agent.py        # Process CSV datasets and generate embeddings
│   ├── vectordb.py      # Manage interaction with Qdrant (upload, search, export)
│   ├── cli.py           # Command Line Interface using Typer
│
├── outputs/             # Generated embeddings (.pkl)
├── exported_data/       # Full collection exports
├── exported_search/     # Search result exports
└── README.md
```

---

## 🧐 Tech Stack

- **Python 3.10+**
- **Typer** – Modern CLI framework
- **Sentence Transformers** – Open-source text embeddings
- **Qdrant** – Vector database (runs locally via Docker)

---

## 🚀 Future Plans

- Tauri + TypeScript Web Inspector (Web-based search & visualization).
- Improved export options (custom filters, advanced queries).
- Support for multi-label categories.
- Authentication for production deployments.

---

## 🕋️ License

This project is open source and free to use under the [MIT License](LICENSE).

---

## 💬 Contact

For feedback, collaboration, or ideas:  
**[LinkedIN](https://www.linkedin.com/in/furkancankaya/)** 
**[GitHub](https://github.com/cankaya96)**

