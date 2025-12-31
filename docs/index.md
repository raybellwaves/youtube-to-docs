# youtube-to-docs
[![PyPI version](https://img.shields.io/pypi/v/youtube-to-docs.svg)](https://pypi.org/project/youtube-to-docs/)

Convert YouTube videos into structured docs, summaries, audio, and visual assets for easier discovery.

## At a glance

- **CLI + MCP server**: run locally or through AI agents.
- **Rich outputs**: transcripts, summaries, Q&A, speakers, audio, and infographics.
- **Structured data**: a CSV you can load into Sheets, Excel, or a database.

## Quickstart

```bash
uvx youtube-to-docs --help
```

Summarize a single video:

```bash
uvx youtube-to-docs atmGAHYpf_c -m gemini-3-flash-preview
```

Install as a Gemini CLI extension:

```bash
gemini extensions install https://github.com/DoIT-Artificial-Intelligence/youtube-to-docs.git
```

## Where to go next

- [Usage guide](usage.md)
- [How it works](how-this-works.md)
- [Development guide](development.md)

## Output location

Artifacts are written to `youtube-to-docs-artifacts/` by default. Use `-o` to set a custom CSV output path.

*Created with the help of AI. All artifacts have been checked and work as expected.*
