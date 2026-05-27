from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence. It is concerned with the interactions between computers 
and human language, in particular how to program computers to process and analyze 
large amounts of natural language data.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=15
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} [{len(chunk)} chars]: {chunk}")

# Output:
# Chunk 1 [96 chars]: Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial
# Chunk 2 [84 chars]: and artificial intelligence. It is concerned with the interactions between computers
# Chunk 3 [81 chars]: and human language, in particular how to program computers to process and analyze
# Chunk 4 [39 chars]: large amounts of natural language data.

# Findings:
# The default seperators are: ["\n\n", "\n", " ", ""]
# The text is split into chunks of 100 characters with an overlap of 15 characters between chunks. The splitter tries to split the text at the specified separators, starting with the longest separator and working down to the shortest. If it cannot find a separator within the chunk size, it will split at the chunk size.