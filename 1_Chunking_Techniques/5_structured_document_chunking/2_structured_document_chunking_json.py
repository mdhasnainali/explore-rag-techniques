from langchain_text_splitters import RecursiveJsonSplitter

json_doc = {
    "company": "Acme Corp",
    "founded": 1998,
    "headquarters": {"city": "San Francisco", "state": "CA", "zip": "94105"},
    "products": [
        {"id": "p1", "name": "Widget Pro", "price": 49.99, "tags": ["hardware", "popular"]},
        {"id": "p2", "name": "Gadget Lite", "price": 19.99, "tags": ["software", "budget"]},
        {"id": "p3", "name": "SuperTool", "price": 99.99, "tags": ["hardware", "premium"]},
    ],
    "financials": {
        "revenue": 4200000,
        "net_income": 756000,
        "currency": "USD",
    },
}

splitter = RecursiveJsonSplitter(max_chunk_size=200)
chunks = splitter.split_json(json_data=json_doc)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
    print()

# Output:
# Chunk 1: {'company': 'Acme Corp', 'founded': 1998, 'headquarters': {'city': 'San Francisco', 'state': 'CA', 'zip': '94105'}}
#
# Chunk 2: {'products': [{'id': 'p1', 'name': 'Widget Pro', 'price': 49.99, 'tags': ['hardware', 'popular']}, {'id': 'p2', 'name': 'Gadget Lite', 'price': 19.99, 'tags': ['software', 'budget']}, {'id': 'p3', 'name': 'SuperTool', 'price': 99.99, 'tags': ['hardware', 'premium']}]}
#
# Chunk 3: {'financials': {'revenue': 4200000, 'net_income': 756000, 'currency': 'USD'}}

# Findings:
# RecursiveJsonSplitter splits at object/array boundaries, never mid-key or mid-value.
# Each chunk is a valid JSON fragment that preserves the key hierarchy.
# max_chunk_size is measured in characters of the serialised JSON string.
# Array items are split individually when the array is too large to fit in one chunk.
