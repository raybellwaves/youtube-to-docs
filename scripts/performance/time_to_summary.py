
import os
import time
from datetime import date
import re

import polars as pl

from youtube_to_docs.llms import generate_summary
from youtube_to_docs.transcript import fetch_transcript

# --- Constants ---
VIDEO_ID = "atmGAHYpf_c"
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
OUTPUT_CSV = "time_to_summarize.csv"
TRANSCRIPT_FILE = "scripts/performance/transcript.vtt.en.vtt"

def parse_vtt(filepath: str) -> str:
    """Parses a VTT file and extracts the transcript text."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    text_lines = []
    for line in lines:
        # Skip VTT metadata
        if "-->" in line or line.strip() in ["WEBVTT", "Kind: captions", "Language: en"]:
            continue
        # Remove timestamps and other VTT tags
        line = re.sub(r"<[^>]+>", "", line)
        # remove align start position
        line = re.sub(r'align:start position:0%', '', line)
        line = line.strip()

        if line:
            text_lines.append(line)

    # Remove duplicate lines while preserving order
    unique_lines = []
    [unique_lines.append(line) for line in text_lines if line not in unique_lines]

    return " ".join(unique_lines)


def main():
    """
    Times how long it takes for various models to summarize a YouTube video,
    then writes the results to a CSV file.
    """
    print(f"Parsing transcript from file: {TRANSCRIPT_FILE}...")

    if not os.path.exists(TRANSCRIPT_FILE):
        print(f"Transcript file not found at {TRANSCRIPT_FILE}. Exiting.")
        return

    transcript = parse_vtt(TRANSCRIPT_FILE)
    print("Transcript parsed successfully.")

    results = []
    today = date.today().isoformat()

    for model in MODELS:
        print(f"Summarizing with model: {model}...")
        start_time = time.monotonic()
        try:
            summary_text, _, _ = generate_summary(
                model_name=model,
                transcript=transcript,
                video_title="Unknown Video",
                url=f"https://www.youtube.com/watch?v={VIDEO_ID}",
            )
            # Check for empty or error responses
            if not summary_text or "Error:" in summary_text:
                print(f"  -> Model {model} returned an error or empty response.")
                duration = 999
            else:
                duration = time.monotonic() - start_time
                print(f"  -> Time taken: {duration:.2f} seconds")
        except Exception as e:
            print(f"  -> An exception occurred while using model {model}: {e}")
            duration = 999

        results.append(
            {"model": model, "time (seconds)": duration, "date (today)": today}
        )

    # Create and sort the DataFrame
    df = pl.DataFrame(results)
    df = df.sort("time (seconds)")

    # Write to CSV
    df.write_csv(OUTPUT_CSV)
    print(f"\nResults written to {OUTPUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()
