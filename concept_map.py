# backend/concept_map.py

from typing import Dict, Any
import ast
import os

from groq import Groq

from .config import LLM_MODEL

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def build_concept_map(full_text: str) -> Dict[str, Any]:
    """
    Ask the LLM to extract a simple concept map.
    Returns dict with 'nodes' and 'edges'.
    """
    prompt = """
Read the following research paper text and extract a simple concept map.

Return a Python dictionary with this exact format:

{
  "nodes": ["Problem", "Method", "Dataset", "Results"],
  "edges": [
    ["Problem", "Method"],
    ["Method", "Results"]
  ]
}

Use 4–8 nodes. Nodes should be brief phrases.
Edges are directional: ["source", "target"].
Only return the dictionary, nothing else.

Paper text:
""" + full_text[:6000]

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You extract structured concept maps from research papers.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content.strip()
    try:
        data = ast.literal_eval(raw)
        if "nodes" in data and "edges" in data:
            return data
    except Exception:
        pass

    # fallback if parsing fails
    return {
        "nodes": ["Paper"],
        "edges": [],
    }
