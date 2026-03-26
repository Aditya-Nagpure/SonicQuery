You are helping build a minimal multimodal RAG system.

Pipeline:
Audio/Video → Whisper → Chunk → Embeddings → FAISS → Retrieve → Claude → Answer

Guidelines:

Write clean, modular Python (no overengineering)
Keep functions small and reusable
Follow existing files (main.py, rag.py, transcribe.py, utils.py)
Always retrieve context before generating answers
Use metadata (timestamps) when possible
Handle edge cases (empty input, long files)

Goal:
Keep the system simple, efficient, and interview-ready

multimodal-analyzer/
│
├── main.py            # end-to-end pipeline (entry point)
├── rag.py             # retrieval + QA logic
├── transcribe.py      # Whisper integration
├── utils.py           # chunking, helpers
│
├── data/              # optional (or ignore in .gitignore)
│
├── requirements.txt
├── README.md
└── .env

🧠 What each file does (1-liner)
main.py → runs full pipeline
transcribe.py → audio/video → text (OpenAI Whisper)
utils.py → chunking + preprocessing
rag.py → embeddings + retrieval + Claude/groq Q&A