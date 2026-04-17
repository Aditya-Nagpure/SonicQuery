import os

from groq import Groq


def summarize(segments: list[dict]) -> str:
    """
    Concatenate all transcript segments and ask Groq for a structured summary.
    """
    transcript = "\n".join(
        f"[{s['start']:.1f}s] {s['text']}" for s in segments
    )

    prompt = f"""You are summarizing a transcript. Structure your response as:

**Summary**
A concise 3–5 sentence overview of the content.

**Key Points**
- Bullet the main topics or ideas covered.

**Notable Moments**
- Call out any specific timestamps worth revisiting, with a short reason.

Transcript:
{transcript}"""

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.3,
    )

    return response.choices[0].message.content
