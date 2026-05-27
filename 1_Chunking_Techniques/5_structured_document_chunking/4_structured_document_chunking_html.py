from langchain_text_splitters import HTMLHeaderTextSplitter

html_doc = """
<!DOCTYPE html>
<html>
<body>
  <h1>Annual Report 2024</h1>

  <h2>Executive Summary</h2>
  <p>Revenue grew 23% YoY driven by cloud segment.</p>
  <p>Operating margin improved to 18%.</p>

  <h2>Financial Performance</h2>
  <p>Total revenue: $4.2B</p>
  <p>Net income: $756M</p>

  <h3>Quarterly Breakdown</h3>
  <p>Q1: $980M | Q2: $1.02B | Q3: $1.1B | Q4: $1.1B</p>

  <h2>Risk Factors</h2>
  <p>Supply chain disruptions remain a concern.</p>
  <p>Regulatory changes in EU may impact operations.</p>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "document_title"),
    ("h2", "section"),
    ("h3", "subsection"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_doc)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}")
    print(f"  Metadata: {chunk.metadata}")
    print(f"  Content:  {chunk.page_content}")
    print()

# Output:
# Chunk 1
#   Metadata: {'document_title': 'Annual Report 2024'}
#   Content:  Annual Report 2024
#
# Chunk 2
#   Metadata: {'document_title': 'Annual Report 2024', 'section': 'Executive Summary'}
#   Content:  Executive Summary
#
# Chunk 3
#   Metadata: {'document_title': 'Annual Report 2024', 'section': 'Executive Summary'}
#   Content:  Revenue grew 23% YoY driven by cloud segment.
#             Operating margin improved to 18%.
#
# Chunk 4
#   Metadata: {'document_title': 'Annual Report 2024', 'section': 'Financial Performance'}
#   Content:  Financial Performance
#
# Chunk 5
#   Metadata: {'document_title': 'Annual Report 2024', 'section': 'Financial Performance'}
#   Content:  Total revenue: $4.2B
#             Net income: $756M
#
# Chunk 6
#   Metadata: {'document_title': 'Annual Report 2024', 'section': 'Financial Performance', 'subsection': 'Quarterly Breakdown'}
#   Content:  Quarterly Breakdown
#
# Chunk 7
#   Metadata: {'document_title': 'Annual Report 2024', 'section': 'Financial Performance', 'subsection': 'Quarterly Breakdown'}
#   Content:  Q1: $980M | Q2: $1.02B | Q3: $1.1B | Q4: $1.1B
#
# Chunk 8
#   Metadata: {'document_title': 'Annual Report 2024', 'section': 'Risk Factors'}
#   Content:  Risk Factors
#
# Chunk 9
#   Metadata: {'document_title': 'Annual Report 2024', 'section': 'Risk Factors'}
#   Content:  Supply chain disruptions remain a concern.
#             Regulatory changes in EU may impact operations.

# Findings:
# HTMLHeaderTextSplitter splits on heading tags (h1–h6) and attaches all ancestor
# headings as metadata on each chunk.
# The content of each chunk is the plain text between the current heading and the next
# heading of equal or higher level — HTML tags are stripped.
# Metadata accumulates: an h3 chunk carries h1, h2, and h3 in its metadata dict.
