import shutil
from functools import lru_cache
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path

# Resolve paths from this file so the app works whether it is started from the
# project root or from the backend folder.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Folder containing the Markdown knowledge base.
KB_PATH = PROJECT_ROOT / "brittany-kb"

# Keep the Chroma database in the same backend/vector_db location you already use.
VECTOR_DB_PATH = BASE_DIR / "vector_db"

# Larger chunks keep related Markdown sections together, which usually gives the
# LLM better context than many tiny fragments.
CHUNK_SIZE = 1400
CHUNK_OVERLAP = 250

# Retrieve a few extra candidates, then use MMR to keep the final context diverse
# and reduce near-duplicate passages without adding another service or framework.
RETRIEVAL_K = 5
MMR_FETCH_K = 12
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_embeddings():
    """Load the embedding model once per process.

    Creating ``HuggingFaceEmbeddings`` loads a sentence-transformers model.  In
    the previous implementation that happened for every chat request, which
    made retrieval unnecessarily slow and repeatedly attempted a network
    download when the model was not already cached.
    """
    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the embedding model. Run create_db.py once with "
            f"network access so '{EMBEDDING_MODEL}' can be downloaded and cached."
        ) from exc


def _source_filename(document):
    if "file_name" in document.metadata:
        return document.metadata["file_name"]

    # Chroma stores the loader's source path in metadata; show only the filename
    # so prompts stay clean and easy for the model to read.
    source = document.metadata.get("source", "unknown.md")
    return Path(source).name


def _dedupe_documents(documents):
    # Overlapping chunks and MMR can still return repeated text. Removing exact
    # duplicates keeps the prompt shorter and prevents repeated context bias.
    seen = set()
    unique_documents = []

    for document in documents:
        normalized_content = " ".join(document.page_content.split())
        duplicate_key = (_source_filename(document), normalized_content)

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)
        unique_documents.append(document)

    return unique_documents


def _format_context(documents):
    # Format each retrieved chunk with its source file so the LLM can ground its
    # answer and the prompt remains readable.
    context_blocks = []

    for document in documents:
        context_blocks.append(
            f"File: {_source_filename(document)}\n"
            "⸻\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n".join(context_blocks)


def create_database():
    if not KB_PATH.is_dir():
        raise FileNotFoundError(f"Knowledge-base folder does not exist: {KB_PATH}")

    # Load all Markdown files
    loader = DirectoryLoader(
        str(KB_PATH), glob="**/*.md", loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No Markdown files found in knowledge base: {KB_PATH}")

    print(f"Loaded {len(documents)} documents")

    # Split Markdown into moderately larger overlapping chunks so related ideas
    # stay together while still giving the retriever enough granularity.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Store the clean filename in metadata during indexing. The search function
    # also has a fallback, so older databases still work until you rebuild them.
    for chunk in chunks:
        chunk.metadata["file_name"] = _source_filename(chunk)

    chunks = _dedupe_documents(chunks)

    if not chunks:
        raise ValueError("The knowledge base did not produce any searchable chunks")

    print(f"Created {len(chunks)} chunks")

    embeddings = _get_embeddings()

    # Rebuild only after loading and splitting succeeds, so a malformed or
    # empty knowledge base cannot destroy a working database.
    if VECTOR_DB_PATH.exists():
        shutil.rmtree(VECTOR_DB_PATH)

    # Create and save the vector database.
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_DB_PATH)
    )

    print("Database created successfully!")


def search_knowledge(question):
    """Return grounded, de-duplicated context relevant to *question*."""
    if not isinstance(question, str) or not question.strip():
        return ""

    if not VECTOR_DB_PATH.is_dir():
        raise FileNotFoundError(
            f"Vector database not found at {VECTOR_DB_PATH}. "
            "Run create_db.py before starting the server."
        )

    embeddings = _get_embeddings()

    # Load the existing vector database
    db = Chroma(
        persist_directory=str(VECTOR_DB_PATH),
        embedding_function=embeddings
    )

    # MMR keeps strong matches while encouraging variety, which helps avoid a
    # prompt filled with several overlapping chunks from the same section.
    collection_count = db._collection.count()
    if collection_count == 0:
        return ""

    result_count = min(RETRIEVAL_K, collection_count)
    fetch_count = min(max(result_count, MMR_FETCH_K), collection_count)
    results = db.max_marginal_relevance_search(
        question.strip(),
        k=result_count,
        fetch_k=fetch_count,
    )

    results = _dedupe_documents(results)

    return _format_context(results)
