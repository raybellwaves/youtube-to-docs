# YouTube-to-Docs

`youtube-to-docs` is a versatile tool designed to convert YouTube content into structured documentation, including transcripts, summaries, audio, and infographics. It is primarily designed as a Command Line Interface (CLI) tool but can also be used as a Python library.

## Features

- **Multimodal Output**: It doesn't just stop at text.
  - **Summaries**: Uses state-of-the-art LLMs to create concise summaries.
  - **Speaker Extraction**: Automatically identifies speakers and their titles/roles from the transcript.
  - **Audio (TTS)**: Converts summaries into audio files, perfect for listening on the go.
  - **Visuals (Infographics)**: Generates AI-created infographics to visually represent the content.
  - **Videos**: Combines infographics and audio into a shareable video summary.
- **Structured Data (CSV/Polars)**:
  - Instead of loose files, metadata and paths are organized into a robust CSV file using `polars`.
  - This makes it incredibly easy to import the data into Google Sheets, Excel, or a database for further analysis or publishing.
- **Batch Processing**:
  - Seamlessly handles individual videos, entire playlists, or full channels with a single command.
- **Multi-Provider Support**:
  - Agnostic to the LLM provider. Whether you use Google Gemini, Vertex AI, AWS Bedrock, or Azure Foundry, you can plug in your preferred model.
- **Cost Awareness**:
  - When using paid API models, it tracks and estimates the cost of summarization, saving it directly to your data file.

## Getting Started

### Prerequisites

- Python 3.14 or higher
- `uv`

### Installation

1. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

To run the script locally using the entry point defined in `pyproject.toml`:

```bash
uv run youtube-to-docs --model gemini-3-flash-preview
```

Alternatively, you can run it as a module:

```bash
uv run python -m youtube_to_docs.main --model gemini-3-flash-preview
```

### Examples

**1. Summarize the default video using a single model:**
```bash
youtube-to-docs -m gemini-3-flash-preview
```

**2. Generate a transcript using Gemini 2.0 Flash and summarize:**
```bash
youtube-to-docs -t gemini-2.0-flash-exp -m gemini-3-flash-preview
```

**3. Process the default video and save to a custom CSV:**
```bash
youtube-to-docs -o my-docs.csv
```

## Documentation

For more detailed information, please refer to the full documentation:

- **[Usage Guide](docs/usage.md)**
- **[How This Works](docs/how-this-works.md)**
- **[Development Guide](docs/development.md)**

## Contributing

We welcome contributions! Please see the [Development Guide](docs/development.md) for more information.

---

*Created with the help of AI. All artifacts have been checked and work as expected.*
