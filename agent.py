import os, json
import openai

PROMPT_TEMPLATE = """SYSTEM:
You are acting as a {persona} conducting an interview.

CONTEXT (verbatim transcript from a prior human interview):
\"\"\"{transcript}\"\"\"

TASK:
Based on the interview above, answer the following survey question **as that same person would answer today**.
Return your answer as JSON with keys:
  "answer": string,
  "confidence": float (0-1)

QUESTION:
{question}
"""

def synthesize_answer(record, question: str, persona: str):
    prompt = PROMPT_TEMPLATE.format(
        persona=persona,
        transcript=record["transcript_text"],
        question=question,
    )

    response = openai.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=128,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    usage = response.usage.total_tokens if hasattr(response, "usage") else 0
    return data["answer"], data.get("confidence", 0.5), usage

