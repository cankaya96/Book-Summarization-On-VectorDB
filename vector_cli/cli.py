import typer
from vector_cli.agent import run_agent
from vector_cli.vectordb import upload_to_qdrant, inspect_qdrant, search_in_qdrant, export_qdrant, clear_qdrant, search_and_export_in_qdrant

app = typer.Typer(help="Vector CLI Tool - Manage your embeddings and collections easily.")

@app.command()
def agent(
    input_file: str = typer.Argument(..., help="Input dataset file (CSV)"),
    text_column: str = typer.Option("Summary", help="Which column contains the text to embed"),
    title_column: str = typer.Option("book_name", help="Which column contains the book titles"),
    category_column: str = typer.Option("categories", help="Which column contains the categories")
):
    """
    Process input dataset and create embeddings, save to pickle.
    """
    run_agent(input_file, text_column=text_column, title_column=title_column, category_column=category_column)

@app.command()
def upload(
    pickle_file: str = typer.Argument(..., help="Pickle file to upload"),
    collection_name: str = typer.Argument("book_summaries", help="Collection name in Qdrant")
):
    """
    Upload embeddings from pickle to Qdrant.
    """
    upload_to_qdrant(pickle_file, collection_name)

@app.command()
def inspect(
    pickle_file: str = typer.Option("outputs/vector_data.pkl", help="Path to the pickle file"),
    collection_name: str = typer.Argument("book_summaries", help="Collection name to inspect"),
    limit: int = typer.Option(5, help="Number of records to show")
):
    """
    Inspect records from a collection in Qdrant.
    """
    inspect_qdrant(pickle_file, collection_name, limit)


@app.command()
def search(
    query: str = typer.Argument(..., help="Query text for vector search"),
    collection_name: str = typer.Argument("book_summaries", help="Collection name to search in"),
    limit: int = typer.Option(5, help="Number of results to return"),
    unique: bool = typer.Option(False, help="Return only unique book names")
):
    """
    Search in Qdrant using a query string.
    """
    search_in_qdrant(query, collection_name, limit, unique)

@app.command()
def export(
    collection_name: str = typer.Argument("book_summaries", help="Collection name to export"),
    format: str = typer.Option("json", help="Export format: json or csv"),
    limit: int = typer.Option(100, help="Number of records to export"),
    output_path: str = typer.Option("exported_data", help="Folder to save exported files")
):
    """
    Export collection records to JSON or CSV file.
    """
    export_qdrant(collection_name, format, limit, output_path)

@app.command()
def clear(
    collection_name: str = typer.Argument("book_summaries", help="Collection name to clear from Qdrant")
):
    """
    Clear (delete) a collection in Qdrant.
    """
    clear_qdrant(collection_name)

@app.command()
def search_export(
    query: str = typer.Argument(..., help="Query text for vector search"),
    collection_name: str = typer.Argument("book_summaries", help="Collection name to search in"),
    limit: int = typer.Option(5, help="Number of results to return"),
    unique: bool = typer.Option(False, help="Return only unique book names"),
    output_path: str = typer.Option("exported_search", help="Folder to save exported search results"),
    format: str = typer.Option("json", help="Export format: json or csv")
):
    """
    Search in Qdrant and export the search results to a file.
    """
    search_and_export_in_qdrant(query, collection_name, limit, unique, output_path, format)


if __name__ == "__main__":
    app()
