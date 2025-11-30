# backend/summarizer.py

import os
from groq import Groq

from .config import LLM_MODEL

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_summary(full_text: str, level: str = "beginner") -> str:
    """
    level: 'beginner' | 'intermediate' | 'expert'
    """
    if level == "beginner":
        style = (
            "Explain the paper in very simple language, "
            "3–5 short bullet points, no heavy math."
        )
    elif level == "intermediate":
        style = (
            "Explain the main problem, method, dataset, and results. "
            "Use correct technical terms but stay concise."
        )
    else:
        style = (
            "Give a compact technical summary focusing on methodology, "
            "assumptions, and limitations. Assume the reader is a CS undergrad."
        )

    prompt = (
        f"{style}\n\n"
        "Here is the paper's text:\n"
        f"{full_text[:8000]}"  # truncate to avoid huge context
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You summarize research papers for students."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return resp.choices[0].message.content
