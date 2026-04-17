# SonicQuery

Turn any audio or video file into a structured summary with key points and timestamps — powered entirely by Groq.

**Pipeline:** Audio/Video → Groq Whisper API → Timestamped Transcript → Groq LLaMA 3.3 70B → Structured Summary

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

---

## Usage

```bash
python main.py path/to/audio.mp3
python main.py path/to/video.mp4
python main.py path/to/podcast.m4a
```

Output:

```
Transcribing: podcast.m4a
  Duration: 315.0 min → splitting into 4 chunk(s)...
  Chunk 1/4: 0.0–90.0 min...
  ...
  Generating summary...

**Summary**
A concise overview of the content...

**Key Points**
- Main topics covered...

**Notable Moments**
- [12.4s] Something worth revisiting...
```

Transcripts are cached in `.transcript_cache/` — re-runs skip transcription entirely and go straight to the summary.

---

## Stack

| Component | Tool | Why |
|-----------|------|-----|
| Transcription | Groq `whisper-large-v3-turbo` | Cloud-based, no GPU needed, handles any file via chunking |
| Summarization | Groq `llama-3.3-70b-versatile` | Fast inference (~300 tok/s), strong reasoning, free tier |
| Audio compression | `imageio-ffmpeg` | Bundled ffmpeg binary — no system install or sudo required |
| Env | `python-dotenv` | Keeps API keys out of code |

---

## Project Structure

```
main.py        # entry point — wires the pipeline
transcribe.py  # Groq Whisper API → [{text, start, end}], with compression + chunking + caching
rag.py         # builds prompt from transcript and calls Groq LLM for summary
utils.py       # overlapping chunk windows with timestamp metadata
```

---

## How it works

1. **Transcribe** — audio/video is uploaded to Groq's Whisper API and returned as timestamped segments
2. **Large file handling** — files over 24 MB are automatically compressed (mono, 16 kHz, 32 kbps) using a bundled ffmpeg binary; files over 90 minutes are split into chunks, each transcribed separately with timestamps offset correctly
3. **Cache** — transcript is saved as JSON so the same file is never re-uploaded
4. **Summarize** — the full timestamped transcript is sent to Groq LLaMA 3.3 70B with a structured prompt requesting a Summary, Key Points, and Notable Moments