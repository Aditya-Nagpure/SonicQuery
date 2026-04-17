import sys
from dotenv import load_dotenv

from transcribe import transcribe
from rag import summarize


def run(file_path: str) -> None:
    load_dotenv()

    print(f"\nTranscribing: {file_path}")
    segments = transcribe(file_path)
    print(f"  {len(segments)} segments transcribed.")

    print("  Generating summary...\n")
    result = summarize(segments)
    print(result)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <audio_or_video_file>")
        sys.exit(1)

    run(sys.argv[1])
