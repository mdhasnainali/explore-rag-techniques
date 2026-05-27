from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code = """
import os
from dataclasses import dataclass


@dataclass
class Config:
    host: str
    port: int
    debug: bool = False


def connect(config: Config) -> bool:
    \"\"\"Establish a connection using the given config.\"\"\"
    if config.debug:
        print(f"Connecting to {config.host}:{config.port}")
    return True


class DatabaseClient:
    def __init__(self, config: Config):
        self.config = config
        self._connected = False

    def open(self):
        self._connected = connect(self.config)

    def query(self, sql: str) -> list:
        if not self._connected:
            raise RuntimeError("Not connected")
        return []

    def close(self):
        self._connected = False
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=20,
)

chunks = splitter.split_text(python_code)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} [{len(chunk)} chars]:")
    print(chunk)
    print()

# Output:
# Chunk 1 [56 chars]:
# import os
# from dataclasses import dataclass
#
#
# @dataclass
#
# Chunk 2 [65 chars]:
# class Config:
#     host: str
#     port: int
#     debug: bool = False
#
# Chunk 3 [190 chars]:
# def connect(config: Config) -> bool:
#     """Establish a connection using the given config."""
#     if config.debug:
#         print(f"Connecting to {config.host}:{config.port}")
#     return True
#
# Chunk 4 [190 chars]:
# class DatabaseClient:
#     def __init__(self, config: Config):
#         self.config = config
#         self._connected = False
#
#     def open(self):
#         self._connected = connect(self.config)
#
# Chunk 5 [186 chars]:
# def query(self, sql: str) -> list:
#         if not self._connected:
#             raise RuntimeError("Not connected")
#         return []
#
#     def close(self):
#         self._connected = False

# Findings:
# from_language() loads Python-specific separators:
#   ["\nclass ", "\ndef ", "\n\tdef ", "\n\n", "\n", " ", ""]
# Splits prefer class and function boundaries first, then fall back to
# blank lines, then lines, then words — same recursive logic as
# RecursiveCharacterTextSplitter but with language-aware separator order.
# Supported languages: python, js, ts, java, go, rust, cpp, html, markdown, and more.
# Check Language enum: from langchain_text_splitters import Language; list(Language)
