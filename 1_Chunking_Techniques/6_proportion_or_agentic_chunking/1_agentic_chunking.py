import json
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

# Initializes the client (looks for OPENAI_API_KEY in your environment variables)
client = OpenAI()

def propositionize(passage: str) -> list[str]:
    prompt = f"""Decompose the following passage into atomic, self-contained 
factual propositions. Each proposition must:
- Express exactly ONE fact
- Be fully understandable without additional context
- Use proper nouns, not pronouns

Passage: {passage}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Highly efficient for structuring text tasks
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        # Forces the model to return a strict JSON schema (a list of strings)
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "propositions_list",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "propositions": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["propositions"],
                    "additionalProperties": False
                }
            }
        }
    )
    
    # Safely parse the guaranteed JSON response
    result = json.loads(response.choices[0].message.content)
    return result["propositions"]

passage = """
Marie Curie, born in Warsaw in 1867, was the first woman to win a Nobel Prize.
She won it twice — in Physics (1903) and Chemistry (1911) — making her the only 
person to win Nobel Prizes in two different sciences.
"""

props = propositionize(passage)
for i, p in enumerate(props):
    print(f"[P{i+1}] {p}")
    

# Output:
# [P1] Marie Curie was born in Warsaw.
# [P2] Marie Curie was born in 1867.
# [P3] Marie Curie was the first woman to win a Nobel Prize.
# [P4] Marie Curie won the Nobel Prize in Physics in 1903.
# [P5] Marie Curie won the Nobel Prize in Chemistry in 1911.
# [P6] Marie Curie is the only person to win Nobel Prizes in two different sciences.