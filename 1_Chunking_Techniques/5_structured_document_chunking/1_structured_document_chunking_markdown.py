from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_doc = """
# Annual Report 2024

## Executive Summary
Revenue grew 23% YoY driven by cloud segment.
Operating margin improved to 18%.

## Financial Performance
Total revenue: $4.2B
Net income: $756M
Free cash flow: $1.1B

## Risk Factors
Supply chain disruptions remain a concern.
Regulatory changes in EU may impact operations.
"""

headers_to_split_on = [
    ("#", "h1_title"),
    ("##", "h2_section"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_doc)

for chunk in chunks:
    print(f"Metadata: {chunk.metadata}")
    print(f"Content:  {chunk.page_content[:80]}...")
    print()
    
    
# Output:
# Metadata: {'h1_title': 'Annual Report 2024', 'h2_section': 'Executive Summary'}
# Content:  Revenue grew 23% YoY driven by cloud segment.
# Operating margin improved to 18%....

# Metadata: {'h1_title': 'Annual Report 2024', 'h2_section': 'Financial Performance'}
# Content:  Total revenue: $4.2B
# Net income: $756M
# Free cash flow: $1.1B...

# Metadata: {'h1_title': 'Annual Report 2024', 'h2_section': 'Risk Factors'}
# Content:  Supply chain disruptions remain a concern.
# Regulatory changes in EU may impact o...