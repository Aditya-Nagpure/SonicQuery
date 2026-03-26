def chunk_segments(segments: list[dict], chunk_size: int = 5, overlap: int = 1) -> list[dict]:
    """
    Group transcript segments into overlapping chunks.

    Each chunk contains `chunk_size` segments, sliding by (chunk_size - overlap).
    Returns a list of: {text, start, end}
    """
    if not segments:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(segments), step):
        group = segments[i : i + chunk_size]
        chunks.append({
            "text": " ".join(seg["text"] for seg in group),
            "start": group[0]["start"],
            "end": group[-1]["end"],
        })

    return chunks
