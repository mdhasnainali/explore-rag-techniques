from langchain_text_splitters import CharacterTextSplitter

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. It is concerned with the interactions between computers 
and human language, in particular how to program computers to process and analyze 
large amounts of natural language data.
"""

splitter = CharacterTextSplitter(
    chunk_size=100,        # characters per chunk
    chunk_overlap=15,      # 15 characters overlap
    separator=""           # Default is "/n/n"
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} [{len(chunk)} chars]: {chunk}")


# Output:
# Chunk 1 [99 chars]: Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial in
# Chunk 2 [100 chars]: d artificial intelligence. It is concerned with the interactions between computers and human langua
# Chunk 3 [100 chars]: nd human language, in particular how to program computers to process and analyze large amounts of n
# Chunk 4 [36 chars]: ge amounts of natural language data.

# Findings:
# separator = "" (Blind Cutting): You take a ruler, measure exactly 100 characters, and cut with scissors—even if you are right in the middle of a word like automa|tic.
# separator = "\n" (Smart Cutting): You only cut at the end of a line or paragraph where someone hit Enter. This keeps your sentences and ideas whole, preventing words from being chopped in half.
# If we use separator then that rule will use for overlap as well.
