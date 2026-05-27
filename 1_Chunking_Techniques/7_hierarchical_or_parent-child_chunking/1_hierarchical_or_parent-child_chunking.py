from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from sentence_transformers import SentenceTransformer
from typing import List

# ---------------------------------------------------
# Local embedding wrapper
# ---------------------------------------------------
class LocalEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]):
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [
            emb.tolist() if hasattr(emb, "tolist")
            else list(map(float, emb))
            for emb in embeddings
        ]

    def embed_query(self, text: str):
        emb = self.model.encode([text], show_progress_bar=False)[0]
        return emb.tolist() if hasattr(emb, "tolist") else list(map(float, emb))


# ---------------------------------------------------
# Raw document
# ---------------------------------------------------
docs = [
    Document(
        page_content=(
            "Our company remote work policy allows flexible hours. "
            "Employees must be online during core hours from 10 AM to 3 PM EST. "
            "A stipend of $500 is provided annually for home office equipment. "
            "Regarding security, all employees must use the company VPN and "
            "enable two-factor authentication on all work accounts."
        ),
        metadata={"source": "policy_doc_2026.txt"}
    )
]

# ---------------------------------------------------
# Chunk splitters
# ---------------------------------------------------
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=5
)

# ---------------------------------------------------
# Show Parent Chunks
# ---------------------------------------------------
print("\n================ PARENT CHUNKS ================\n")

parent_docs = parent_splitter.split_documents(docs)

for i, doc in enumerate(parent_docs):
    print(f"\n--- Parent Chunk {i + 1} ---")
    print(doc.page_content)

# ---------------------------------------------------
# Show Child Chunks
# ---------------------------------------------------
print("\n================ CHILD CHUNKS ================\n")

for i, parent_doc in enumerate(parent_docs):
    child_docs = child_splitter.split_documents([parent_doc])

    print(f"\n######## Children from Parent {i + 1} ########")

    for j, child in enumerate(child_docs):
        print(f"\nChild Chunk {j + 1}")
        print(child.page_content)

# ---------------------------------------------------
# Vector DB + Store
# ---------------------------------------------------
vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=LocalEmbeddings()
)

store = InMemoryStore()

# ---------------------------------------------------
# Parent Retriever
# ---------------------------------------------------
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# ---------------------------------------------------
# Add docs
# ---------------------------------------------------
retriever.add_documents(docs)

# ---------------------------------------------------
# Query
# ---------------------------------------------------
query = "How much money do I get for my home office setup?"

retrieved_docs = retriever.invoke(query)

print("\n================ QUERY RESULT ================\n")

print(f"Query: {query}\n")
print(f"Number of documents returned: {len(retrieved_docs)}")

print("\n--- Retrieved Parent Document ---\n")
print(retrieved_docs[0].page_content)


# Output:
# ================ PARENT CHUNKS ================


# --- Parent Chunk 1 ---
# Our company remote work policy allows flexible hours. Employees must be online during core hours from 10 AM to 3 PM EST. A stipend of $500 is provided annually for home office equipment. Regarding

# --- Parent Chunk 2 ---
# Regarding security, all employees must use the company VPN and enable two-factor authentication on all work accounts.

# ================ CHILD CHUNKS ================


# ######## Children from Parent 1 ########

# Child Chunk 1
# Our company remote work policy allows flexible

# Child Chunk 2
# hours. Employees must be online during core hours

# Child Chunk 3
# from 10 AM to 3 PM EST. A stipend of $500 is

# Child Chunk 4
# is provided annually for home office equipment.

# Child Chunk 5
# Regarding

# ######## Children from Parent 2 ########

# Child Chunk 1
# Regarding security, all employees must use the

# Child Chunk 2
# the company VPN and enable two-factor

# Child Chunk 3
# authentication on all work accounts.
# Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 15326.14it/s]

# ================ QUERY RESULT ================

# Query: How much money do I get for my home office setup?

# Number of documents returned: 1

# --- Retrieved Parent Document ---

# Our company remote work policy allows flexible hours. Employees must be online during core hours from 10 AM to 3 PM EST. A stipend of $500 is provided annually for home office equipment. Regarding