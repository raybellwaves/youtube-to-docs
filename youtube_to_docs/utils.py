import re

import polars as pl


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
        "Transcript characters from youtube",
        "Audio File",
    ]

    # Filter base_order to only include columns that actually exist
    final_order = [c for c in base_order if c in cols]

    # Add other Transcript characters columns
    other_transcript_chars = [
        c
        for c in cols
        if c.startswith("Transcript characters from ") and c not in final_order
    ]
    final_order.extend(sorted(other_transcript_chars))

    # Add Transcript File columns
    transcript_files = [c for c in cols if c.startswith("Transcript File ")]
    final_order.extend(sorted(transcript_files))

    # Add STT Cost columns
    stt_costs = [c for c in cols if " STT cost" in c]
    final_order.extend(sorted(stt_costs))

    # Add Summary File columns
    summary_files = [
        c for c in cols if c.startswith("Summary File ") and "from youtube" not in c
    ]
    final_order.extend(sorted(summary_files))

    # Add Summary Infographic File columns
    infographic_files = [c for c in cols if c.startswith("Summary Infographic File ")]
    final_order.extend(sorted(infographic_files))

    # Add Summary Infographic Cost columns
    infographic_costs = [c for c in cols if c.startswith("Summary Infographic Cost ")]
    final_order.extend(sorted(infographic_costs))

    # Add Audio File columns (from TTS)
    audio_files = [c for c in cols if c.startswith("Summary Audio File ")]
    final_order.extend(sorted(audio_files))

    # Add Video File columns
    video_files = [c for c in cols if c == "Video File"]
    final_order.extend(video_files)

    # Add QA File columns
    qa_files = [c for c in cols if c.startswith("QA File ")]
    final_order.extend(sorted(qa_files))

    # Add Speakers columns
    speakers = [
        c
        for c in cols
        if c.startswith("Speakers ") and not c.startswith("Speakers File ")
    ]
    final_order.extend(sorted(speakers))

    # Add Speakers File columns
    speakers_files = [c for c in cols if c.startswith("Speakers File ")]
    final_order.extend(sorted(speakers_files))

    # Add Speaker Extraction Cost columns
    speaker_costs = [c for c in cols if " Speaker extraction cost " in c]
    final_order.extend(sorted(speaker_costs))

    # Add Summary Text columns
    summary_texts = [
        c for c in cols if c.startswith("Summary Text ") and "from youtube" not in c
    ]
    final_order.extend(sorted(summary_texts))

    # Add Summary Cost columns
    summary_costs = [
        c for c in cols if " summary cost " in c and "from youtube" not in c
    ]
    final_order.extend(sorted(summary_costs))

    # Add YouTube Summary Text columns (secondary)
    yt_summary_texts = [c for c in cols if "Summary Text" in c and "from youtube" in c]
    final_order.extend(sorted(yt_summary_texts))

    # Add YouTube Summary File columns (secondary)
    yt_summary_files = [c for c in cols if "Summary File" in c and "from youtube" in c]
    final_order.extend(sorted(yt_summary_files))

    # Add YouTube Summary Cost columns (secondary)
    yt_summary_costs = [c for c in cols if "summary cost" in c and "from youtube" in c]
    final_order.extend(sorted(yt_summary_costs))

    # Add QA Text columns
    qa_texts = [c for c in cols if c.startswith("QA Text ")]
    final_order.extend(sorted(qa_texts))

    # Add QA Cost columns
    qa_costs = [c for c in cols if " QA cost " in c]
    final_order.extend(sorted(qa_costs))

    # Add any remaining columns that weren't caught
    remaining = [c for c in cols if c not in final_order]
    final_order.extend(remaining)

    return df.select(final_order)


def normalize_model_name(model_name: str) -> str:
    """
    Normalizes a model name by stripping prefixes and suffixes.
    Suffixes handled: @20251001, -20251001-v1, -v1.
    Prefixes handled: vertex-, bedrock-, foundry-.
    """
    # Strip prefixes
    normalized = model_name
    prefixes = ["vertex-", "bedrock-", "foundry-"]
    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    # Strip suffixes using regex: (@\d{8}|-\d{8}-v\d+|-v\d+)$
    normalized = re.sub(r"(@\d{8}|-\d{8}-v\d+|-v\d+)$", "", normalized)

    return normalized


def add_question_numbers(markdown_table: str) -> str:
    """
    Adds a 'question number' column to the markdown table.
    """
    lines = markdown_table.strip().split("\n")
    if not lines:
        return markdown_table

    # Check if it's a valid table (has header and separator)
    if len(lines) < 2:
        return markdown_table

    new_lines = []
    question_counter = 1

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        if i == 0:
            # Header row
            # Ensure it starts with | (some LLMs might miss it)
            if not stripped_line.startswith("|"):
                stripped_line = "|" + stripped_line
            new_lines.append(f"| question number {stripped_line}")
        elif i == 1 and ("---" in stripped_line or "-|-" in stripped_line):
            # Separator row
            if not stripped_line.startswith("|"):
                stripped_line = "|" + stripped_line
            new_lines.append(f"|---{stripped_line}")
        else:
            # Data row
            if stripped_line.startswith("|"):
                new_lines.append(f"| {question_counter} {stripped_line}")
                question_counter += 1
            else:
                # Handle potential malformed table rows or text outside the table
                if "|" in stripped_line:  # It has columns but maybe missing start pipe
                    new_lines.append(f"| {question_counter} | {stripped_line}")
                    question_counter += 1
                else:
                    new_lines.append(line)

    return "\n".join(new_lines)
