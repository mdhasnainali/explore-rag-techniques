import tiktoken
from langchain_text_splitters import CharacterTextSplitter

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. It is concerned with the interactions between computers 
and human language, in particular how to program computers to process and analyze 
large amounts of natural language data.
"""

splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=100,
    chunk_overlap=0,
     separator=""
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} [{len(chunk)} chars]: {chunk}")

# Count tokens in each chunk to verify
encoding = tiktoken.get_encoding("cl100k_base")

for i, chunk in enumerate(chunks):
    chunk_token_count = len(encoding.encode(chunk))
    print(f"Chunk {i+1} token count: {chunk_token_count}")

# Output:
# Chunk 1 [99 chars]: Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial in
# Chunk 2 [100 chars]: telligence. It is concerned with the interactions between computers and human language, in particul
# Chunk 3 [91 chars]: ar how to program computers to process and analyze large amounts of natural language data.
# Chunk 1 token count: 21
# Chunk 2 token count: 18
# Chunk 3 token count: 17


# Findings:
# There are multiple alternatives of this langchain_text_splitters: spaCy, SenetenceTransformers, NLTK, KoNLPY
