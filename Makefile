.PHONY: install corpus ingest ask test eval benchmark serve docker-build docker-up clean

install:
	pip install -r requirements.txt

corpus:
	python scripts/download_corpus.py

ingest:
	python cli.py ingest data/docs

ask:
	@python cli.py ask "$(Q)"

test:
	pytest -q

eval:
	python eval/evaluate.py

benchmark:
	python scripts/benchmark.py

goldenset:
	python scripts/generate_goldenset.py --per-doc 2 --max-chunks 80

serve:
	uvicorn app.api:app --reload

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf .chroma .pytest_cache __pycache__ */__pycache__ eval/report.json benchmark_results.*
