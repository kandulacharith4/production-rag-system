# Operations Guide

## Running the system

1. Install dependencies: `pip install -r requirements.txt`
2. Set `ANTHROPIC_API_KEY` in your environment.
3. Ingest documents: `python cli.py ingest data/docs`
4. Ask a question: `python cli.py ask "What embedding model is used?"`
5. Or run the API: `uvicorn app.api:app --reload` and POST to `/ask`.

## CI gate

Every pull request triggers `.github/workflows/eval.yml`, which ingests the docs,
runs unit tests, and then runs the golden evaluation. If the mean faithfulness
score is below the configured threshold the build fails and the PR cannot merge.

## Configuration

All tunable parameters (chunk size, top-k, fusion weight, models, threshold) live
in `config.yaml`. Prompts live in `prompts/prompts.yaml` and are versioned.
