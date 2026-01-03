import time
from datetime import date
from pathlib import Path

import polars as pl

from youtube_to_docs.llms import generate_summary
from youtube_to_docs.transcript import fetch_transcript

VIDEO_ID = "atmGAHYpf_c"
VIDEO_URL = f"https://www.youtube.com/watch?v={VIDEO_ID}"
VIDEO_TITLE = VIDEO_ID

MODELS = [
    "bedrock-nova-2-lite-v1",
    "bedrock-nova-pro-v1",
    "bedrock-nova-premier-v1",
    "bedrock-claude-haiku-4-5-20251001-v1",
    "bedrock-claude-sonnet-4-5-20250929-v1",
    "bedrock-claude-opus-4-5-20251101-v1",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "vertex-claude-haiku-4-5@20251001",
    "vertex-claude-sonnet-4-5@20250929",
    "vertex-claude-opus-4-5@20251101",
    "foundry-gpt-5-nano",
    "foundry-gpt-5-mini",
    "foundry-gpt-5",
    "foundry-gpt-5-pro",
    "foundry-gpt-5.2",
    "foundry-gpt-5.2-pro",
]


def get_transcript() -> str:
    result = fetch_transcript(VIDEO_ID, language="en")
    if not result:
        raise RuntimeError(f"Transcript not available for video {VIDEO_ID}")
    transcript, _ = result
    if not transcript.strip():
        raise RuntimeError(f"Empty transcript returned for video {VIDEO_ID}")
    return transcript


def _summary_failed(summary_text: str) -> bool:
    normalized = summary_text.strip().lower()
    if not normalized:
        return True
    return normalized.startswith("error") or "error:" in normalized


def time_model_summary(model_name: str, transcript: str) -> float:
    start = time.perf_counter()
    try:
        summary_text, _, _ = generate_summary(
            model_name, transcript, VIDEO_TITLE, VIDEO_URL, language="en"
        )
    except Exception:
        return 999.0
    end = time.perf_counter()
    if _summary_failed(summary_text):
        return 999.0
    return end - start


def main() -> None:
    transcript = get_transcript()
    today = date.today().isoformat()
    results = []

    for model in MODELS:
        elapsed = time_model_summary(model, transcript)
        results.append(
            {
                "model": model,
                "time (seconds)": f"{elapsed:.3f}",
                "date (today)": today,
            }
        )

    results.sort(key=lambda row: float(row["time (seconds)"]))

    output_path = Path(__file__).resolve().parent / "time_to_summarize.csv"
    pl.DataFrame(results).write_csv(output_path)

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
