import whisper


def transcribe(file_path: str, model_size: str = "base") -> list[dict]:
    """
    Transcribe an audio/video file using Whisper.

    Returns a list of segments: [{text, start, end}, ...]
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(file_path)

    segments = [
        {
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
        }
        for seg in result["segments"]
        if seg["text"].strip()
    ]

    if not segments:
        raise ValueError(f"No speech detected in: {file_path}")

    return segments
