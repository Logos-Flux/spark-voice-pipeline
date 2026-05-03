# Contributing

Thanks for your interest. This is a small project; contributions are
welcome but please keep them focused.

## Before opening a PR

- For non-trivial changes, open an issue first to discuss the approach.
- Run the linter and formatter:
  ```bash
  ruff check .
  black .
  ```
- Add a CHANGELOG.md entry under "Unreleased" describing user-visible
  changes.
- If your change touches the security posture (bind defaults,
  authentication, request handling), call that out explicitly in the PR
  description so reviewers can scrutinize it.

## Development setup

```bash
git clone https://github.com/Logos-Flux/spark-voice-pipeline.git
cd spark-voice-pipeline
pip install -r requirements.txt
pip install ruff black
```

The pipeline depends on external services (Whisper.cpp server, Ollama,
VibeVoice). See the README for setup. Local-only changes that don't
require running the full stack are easiest to land.

## Reporting bugs

Use [GitHub issues](https://github.com/Logos-Flux/spark-voice-pipeline/issues).
For security issues, see [SECURITY.md](SECURITY.md).
