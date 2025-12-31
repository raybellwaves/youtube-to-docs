# youtube-to-docs
[![PyPI version](https://img.shields.io/pypi/v/youtube-to-docs.svg)](https://pypi.org/project/youtube-to-docs/)

Convert YouTube videos into structured docs, summaries, audio, and visual assets for easier discovery.

## Why this exists

`youtube-to-docs` turns videos into a bundle of searchable artifacts: transcripts, summaries, Q&A, audio narration, and infographics. It works as both a CLI and an MCP server for AI agents.

## Quickstart

```bash
uvx youtube-to-docs --help
```

Process a single video with a summary model:

```bash
uvx youtube-to-docs atmGAHYpf_c -m gemini-3-flash-preview
```

Install as a Gemini CLI extension:

```bash
gemini extensions install https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs.git
```

## What you get

- **Structured CSV output** with metadata, file paths, and estimated costs.
- **Transcripts** from YouTube or AI speech-to-text models.
- **Summaries, speaker extraction, and Q&A** via LLMs.
- **Audio summaries and infographics** (optional), plus combined video summaries.

## Repository layout

- `youtube_to_docs/` — CLI and library implementation.
- `docs/` — Usage guides, architecture, and development notes.
- `tests/` — Automated test suite.

## Documentation

- [Usage guide](docs/usage.md)
- [How it works](docs/how-this-works.md)
- [Development guide](docs/development.md)

## Notes

Artifacts are written to `youtube-to-docs-artifacts/` by default. Use `-o` to change the CSV output path.

*Created with the help of AI. All artifacts have been checked and work as expected.*
