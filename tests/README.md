# Tests

This directory contains the test suite for `youtube-to-docs`.

## Manual Tests

The following tests run outside of the standard test suite (they are skipped in CI) as they require manual setup and external credentials:

- `tests/test_workspace.py`: Verifies Google Drive/Workspace integration.
- `tests/test_sharepoint.py`: Verifies OneDrive/SharePoint integration.

### How to Run

To run these tests, you must have the required credentials configured and execute them directly:

```bash
uv run tests/test_workspace.py
uv run tests/test_sharepoint.py
```
