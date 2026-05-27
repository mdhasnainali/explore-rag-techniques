from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# 1. Define your text
text = """The Eiffel Tower is located in Paris.
It was built in 1889 for the World's Fair.
Photosynthesis converts sunlight into energy."""

# 2. Load the raw sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 3. Create a custom wrapper so LangChain can interface with SentenceTransformers
class HuggingFaceSentenceEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Convert embeddings to standard float lists
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


# Wrap your model
langchain_embeddings = HuggingFaceSentenceEmbeddings(model)

# 4. Initialize the SemanticChunker with the wrapper
splitter = SemanticChunker(
    embeddings=langchain_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=20,
)

# 5. Split and print chunks
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1} [{len(chunk)} chars]: {chunk}\n")

# Output:
# Chunk 1 [80 chars]: The Eiffel Tower is located in Paris. It was built in 1889 for the World's Fair.

# Chunk 2 [45 chars]: Photosynthesis converts sunlight into energy.
