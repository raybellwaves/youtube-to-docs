# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "google-auth>=2.45.0",
#     "google-genai>=1.56.0",
#     "google-api-python-client>=2.187.0",
#     "isodate>=0.7.2",
#     "openai>=1.56.0",
#     "polars>=1.36.1",
#     "requests>=2.32.5",
#     "youtube-transcript-api>=1.2.3"
# ///
import argparse
import os
import re
import time

import polars as pl

from youtube_to_docs.infographic import generate_infographic
from youtube_to_docs.llms import generate_summary, get_model_pricing
from youtube_to_docs.transcript import (
    fetch_transcript,
    get_video_details,
    get_youtube_service,
    resolve_video_ids,
)
from youtube_to_docs.tts import process_tts


def reorder_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Reorder columns according to the specified logical structure."""
    cols = df.columns
    base_order = [
        "URL",
        "Title",
        "Description",
        "Data Published",
        "Channel",
        "Tags",
        "Duration",
        "Transcript characters",
    ]

    # Filter base_order to only include columns that actually exist
    final_order = [c for c in base_order if c in cols]

    # Add Transcript File columns
    transcript_files = [c for c in cols if c.startswith("Transcript File ")]
    final_order.extend(sorted(transcript_files))

    # Add Summary File columns
    summary_files = [c for c in cols if c.startswith("Summary File ")]
    final_order.extend(sorted(summary_files))

    # Add Summary Infographic File columns
    infographic_files = [c for c in cols if c.startswith("Summary Infographic File ")]
    final_order.extend(sorted(infographic_files))

    # Add Audio File columns (from TTS)
    audio_files = [c for c in cols if c.startswith("Summary Audio File ")]
    final_order.extend(sorted(audio_files))

    # Add Summary Cost columns
    summary_costs = [c for c in cols if c.endswith(" summary cost ($)")]
    final_order.extend(sorted(summary_costs))

    # Add Summary Text columns
    summary_texts = [c for c in cols if c.startswith("Summary Text ")]
    final_order.extend(sorted(summary_texts))

    # Add any remaining columns that weren't caught
    remaining = [c for c in cols if c not in final_order]
    final_order.extend(remaining)

    return df.select(final_order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "video_id",
        nargs="?",
        default="atmGAHYpf_c",
        help=(
            "Can be one of: \n"
            "A Video ID e.g. 'atmGAHYpf_c'\n"
            "Playlist ID (starts with PL e.g. 'PL8ZxoInteClyHaiReuOHpv6Z4SPrXtYtW')\n"
            "Channel Handle (starts with @ e.g. '@mga-hgo1740')\n"
            "Comma-separated list of Video IDs. (e.g. 'KuPc06JgI_A,GalhDyf3F8g')"
        ),
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="youtube-docs.csv",
        help=("Can be one of: \nLocal file path to save the output CSV file."),
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=(
            "The LLM to use for summarization. Can be one of: \n"
            "Gemini model (e.g., 'gemini-3-flash-preview')\n"
            "GCP Vertex model (prefixed with 'vertex-'). e.g. "
            "vertex-claude-haiku-4-5@20251001\n"
            "AWS Bedrock model (prefixed with 'bedrock-'). e.g. "
            "bedrock-claude-haiku-4-5-20251001-v1\n"
            "Azure Foundry model (prefix with 'foundry-). e.g. 'foundry-gpt-5-mini'\n"
            "Defaults to None."
        ),
    )
    parser.add_argument(
        "--tts",
        default=None,
        help=(
            "The TTS model and voice to use. "
            "Format: {model}-{voice} e.g. 'gemini-2.5-flash-preview-tts-Kore'"
        ),
    )
    parser.add_argument(
        "--infographic",
        default=None,
        help=(
            "The image model to use for generating an infographic. "
            "e.g. 'gemini-2.5-flash-image'"
        ),
    )

    args = parser.parse_args()
    video_id_input = args.video_id
    outfile = args.outfile
    model_name = args.model
    tts_arg = args.tts
    infographic_arg = args.infographic

    youtube_service = get_youtube_service()

    video_ids = resolve_video_ids(video_id_input, youtube_service)

    # Setup Output Directories
    output_dir = os.path.dirname(outfile)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    base_dir = output_dir if output_dir else "."
    transcripts_dir = os.path.join(base_dir, "transcript-files")
    summaries_dir = os.path.join(base_dir, "summary-files")
    infographics_dir = os.path.join(base_dir, "infographic-files")
    os.makedirs(transcripts_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(infographics_dir, exist_ok=True)

    # Load existing CSV if it exists
    existing_df = None
    if os.path.exists(outfile):
        try:
            existing_df = pl.read_csv(outfile)
            print(f"Loaded existing data from {outfile} ({len(existing_df)} rows)")
        except Exception as e:
            print(f"Warning: Could not read existing CSV {outfile}: {e}")

    print(f"Processing {len(video_ids)} videos.")
    print(f"Processing Videos: {video_ids}")
    print(f"Saving to: {outfile}")
    if model_name:
        print(f"Summarizing using model: {model_name}")

    rows = []
    summary_col_name = f"Summary Text {model_name}" if model_name else "Summary Text"
    summary_file_col_name = (
        f"Summary File {model_name}" if model_name else "Summary File"
    )
    summary_cost_col_name = (
        f"{model_name} summary cost ($)" if model_name else "summary cost ($)"
    )
    infographic_col_name = (
        f"Summary Infographic File {model_name} {infographic_arg}"
        if model_name
        else f"Summary Infographic File None {infographic_arg}"
    )

    for video_id in video_ids:
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"Processing Video ID: {video_id}")

        # Check if video already exists in CSV
        existing_row = None
        if existing_df is not None and "URL" in existing_df.columns:
            matches = existing_df.filter(pl.col("URL") == url)
            if not matches.is_empty():
                existing_row = matches.to_dicts()[0]

        # Determine if we need to process this video
        needs_details = existing_row is None
        needs_transcript = existing_row is None
        needs_summary = model_name is not None and (
            existing_row is None
            or summary_col_name not in existing_row
            or not existing_row[summary_col_name]
        )
        needs_infographic = infographic_arg is not None and (
            existing_row is None
            or infographic_col_name not in existing_row
            or not existing_row[infographic_col_name]
        )

        if (
            not needs_details
            and not needs_transcript
            and not needs_summary
            and not needs_infographic
        ):
            print(
                f"Skipping {video_id}: already exists in table with metadata "
                "and summary."
            )
            continue

        # Get Details
        if needs_details:
            details = get_video_details(video_id, youtube_service)
            if not details:
                continue
            (
                video_title,
                description,
                publishedAt,
                channelTitle,
                tags,
                video_duration,
                _,
            ) = details
        else:
            assert existing_row is not None
            video_title = existing_row["Title"]
            description = existing_row["Description"]
            publishedAt = existing_row["Data Published"]
            channelTitle = existing_row["Channel"]
            tags = existing_row["Tags"]
            video_duration = existing_row["Duration"]

        print(f"Video Title: {video_title}")

        # Fetch/Save Transcript
        transcript = ""
        transcript_full_path = ""
        is_generated = False

        # Check existing row for transcript
        if existing_row:
            col_youtube = "Transcript File youtube generated"
            col_human = "Transcript File human generated"
            found_col = None

            if col_youtube in existing_row and existing_row[col_youtube]:
                found_col = col_youtube
                is_generated = True
            elif col_human in existing_row and existing_row[col_human]:
                found_col = col_human
                is_generated = False

            if found_col:
                transcript_full_path = existing_row[found_col]
                if os.path.exists(transcript_full_path):
                    print(
                        f"Reading existing transcript from file: {transcript_full_path}"
                    )
                    with open(transcript_full_path, "r", encoding="utf-8") as f:
                        transcript = f.read()

        if not transcript:
            result = fetch_transcript(video_id)
            if not result:
                continue
            transcript, is_generated = result

            # Save Transcript
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", video_title).replace("\n", " ")
            safe_title = safe_title.replace("\r", "")
            prefix = "youtube generated - " if is_generated else "human generated - "
            transcript_filename = f"{prefix}{video_id} - {safe_title}.txt"
            transcript_full_path = os.path.abspath(
                os.path.join(transcripts_dir, transcript_filename)
            )
            try:
                with open(transcript_full_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
                print(f"Saved transcript: {transcript_filename}")
            except OSError as e:
                print(f"Error writing transcript: {e}")

        # Summarize
        summary_text = existing_row.get(summary_col_name, "") if existing_row else ""
        summary_full_path = (
            existing_row.get(summary_file_col_name, "") if existing_row else ""
        )

        if needs_summary:
            print(f"Summarizing using model: {model_name}")
            summary_text, input_tokens, output_tokens = generate_summary(
                model_name, transcript, video_title, url
            )

            summary_cost = float("nan")
            if model_name:
                input_price, output_price = get_model_pricing(model_name)
                if input_price is not None and output_price is not None:
                    summary_cost = (input_tokens / 1_000_000) * input_price + (
                        output_tokens / 1_000_000
                    ) * output_price
                    print(f"Summary cost: ${summary_cost:.6f}")

            if summaries_dir and summary_text:
                safe_title = re.sub(r"[\\/*?:\"<>|]", "_", video_title).replace(
                    "\n", " "
                )
                safe_title = safe_title.replace("\r", "")
                summary_filename = (
                    f"{model_name} - {video_id} - {safe_title} - summary.md"
                )
                summary_full_path = os.path.abspath(
                    os.path.join(summaries_dir, summary_filename)
                )
                try:
                    with open(summary_full_path, "w", encoding="utf-8") as f:
                        f.write(summary_text)
                    print(f"Saved summary: {summary_filename}")
                except OSError as e:
                    print(f"Error writing summary: {e}")

        # Prepare row data base
        row = existing_row.copy() if existing_row else {}

        if is_generated:
            transcript_col_name = "Transcript File youtube generated"
        else:
            transcript_col_name = "Transcript File human generated"

        row.update(
            {
                "URL": url,
                "Title": video_title,
                "Description": description,
                "Data Published": publishedAt,
                "Channel": channelTitle,
                "Tags": tags,
                "Duration": video_duration,
                "Transcript characters": len(transcript),
                transcript_col_name: transcript_full_path,
            }
        )

        if needs_summary:
            row[summary_file_col_name] = summary_full_path
            row[summary_col_name] = summary_text
            row[summary_cost_col_name] = summary_cost

        # Infographic Generation
        if infographic_arg:
            summary_targets = []

            # 1. Target the specific model requested (if any)
            if model_name:
                col = f"Summary Text {model_name}"
                summary_targets.append((col, model_name, summary_text))

            # 2. Target ALL existing summaries if no model specified
            if not model_name and existing_row:
                for k in existing_row.keys():
                    if k.startswith("Summary Text ") and existing_row[k]:
                        m_name = k[len("Summary Text ") :]
                        summary_targets.append((k, m_name, existing_row[k]))

            for sum_col, m_name, s_text in summary_targets:
                if not s_text:
                    continue

                info_col = f"Summary Infographic File {m_name} {infographic_arg}"

                # Check if already exists
                if info_col in row and row[info_col]:
                    continue

                print(f"Generating infographic for model: {m_name}")
                image_bytes = generate_infographic(infographic_arg, s_text, video_title)
                if image_bytes:
                    safe_title = re.sub(r"[\\/*?:\"<>|]", "_", video_title).replace(
                        "\n", " "
                    )
                    safe_title = safe_title.replace("\r", "")
                    infographic_filename = (
                        f"{m_name} - {infographic_arg} - {video_id} - "
                        f"{safe_title} - infographic.png"
                    )
                    infographic_full_path = os.path.abspath(
                        os.path.join(infographics_dir, infographic_filename)
                    )
                    try:
                        with open(infographic_full_path, "wb") as f:
                            f.write(image_bytes)
                        print(f"Saved infographic: {infographic_filename}")
                        row[info_col] = infographic_full_path
                    except OSError as e:
                        print(f"Error writing infographic: {e}")

        rows.append(row)
        time.sleep(1)

    final_df = None

    if rows:
        new_df = pl.DataFrame(rows)
        if existing_df is not None:
            processed_urls = new_df["URL"].to_list()
            existing_remaining = existing_df.filter(
                ~pl.col("URL").is_in(processed_urls)
            )
            final_df = pl.concat([existing_remaining, new_df], how="diagonal")
        else:
            final_df = new_df
    elif existing_df is not None:
        final_df = existing_df

    if final_df is not None and not final_df.is_empty():
        should_save = bool(rows)

        if "Data Published" in final_df.columns:
            final_df = final_df.sort("Data Published", descending=True)

        if tts_arg:
            print("Checking for TTS generation...")
            final_df = process_tts(final_df, tts_arg, base_dir)
            should_save = True

        if should_save:
            final_df = reorder_columns(final_df)
            final_df.write_csv(outfile)
            print(f"Successfully wrote {len(final_df)} rows to {outfile}")
        else:
            print("No new data to gather or all videos already processed.")
    else:
        print("No new data to gather or all videos already processed.")


if __name__ == "__main__":
    main()
