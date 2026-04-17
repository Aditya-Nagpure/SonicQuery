import json
import os
import re
import subprocess
import tempfile

import imageio_ffmpeg
from groq import Groq


_CACHE_DIR = ".transcript_cache"
_CHUNK_SECONDS = 90 * 60  # 90 min chunks → ~21 MB each at 32 kbps mono


def _cache_path(file_path: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(_CACHE_DIR, f"{base}.json")


def _ffmpeg() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _get_duration(file_path: str) -> float:
    """Return audio/video duration in seconds."""
    result = subprocess.run(
        [_ffmpeg(), "-i", file_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", result.stderr.decode())
    if not match:
        raise ValueError(f"Could not read duration of: {file_path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def _extract_chunk(file_path: str, start: float, duration: float, out_path: str) -> None:
    """Extract a time slice and compress to mono mp3."""
    subprocess.run(
        [
            _ffmpeg(), "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", file_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "32k",
            out_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _transcribe_file(client: Groq, file_path: str, offset: float) -> list[dict]:
    """Send a single file to Groq and return segments with adjusted timestamps."""
    with open(file_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), f),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return [
        {
            "text": seg.text.strip(),
            "start": round(seg.start + offset, 2),
            "end": round(seg.end + offset, 2),
        }
        for seg in response.segments
        if seg.text.strip()
    ]


def transcribe(file_path: str) -> list[dict]:
    """
    Transcribe via Groq's Whisper API.
    Splits large files into 90-minute chunks automatically.
    Caches results so re-runs are instant.
    Returns [{text, start, end}, ...]
    """
    cache = _cache_path(file_path)
    if os.path.exists(cache):
        print("  (loaded from cache)")
        with open(cache) as f:
            return json.load(f)

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    duration = _get_duration(file_path)
    total_chunks = int(duration / _CHUNK_SECONDS) + 1
    print(f"  Duration: {duration/60:.1f} min → splitting into {total_chunks} chunk(s)...")

    all_segments: list[dict] = []
    offset = 0.0
    chunk_num = 1

    while offset < duration:
        chunk_len = min(_CHUNK_SECONDS, duration - offset)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()

        try:
            print(f"  Chunk {chunk_num}/{total_chunks}: {offset/60:.1f}–{(offset+chunk_len)/60:.1f} min...")
            _extract_chunk(file_path, offset, chunk_len, tmp.name)
            segments = _transcribe_file(client, tmp.name, offset)
            all_segments.extend(segments)
        finally:
            os.unlink(tmp.name)

        offset += _CHUNK_SECONDS
        chunk_num += 1

    if not all_segments:
        raise ValueError(f"No speech detected in: {file_path}")

    with open(cache, "w") as f:
        json.dump(all_segments, f)

    return all_segments
