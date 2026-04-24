"""Simple CLI: python cli.py ingest data/docs | python cli.py ask "question" """
import argparse
import json

from app.generate import answer_question
from app.ingest import ingest


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest docs into the vector store")
    p_ing.add_argument("path", help="Directory of documents")

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")

    args = p.parse_args()

    if args.cmd == "ingest":
        print(json.dumps(ingest(args.path), indent=2))
    elif args.cmd == "ask":
        ans = answer_question(args.question)
        print(ans.text)
        if ans.citations:
            print("\nSources:")
            for i in ans.citations:
                c = ans.chunks[i - 1]
                print(f"  [{i}] {c.source} (chunk {c.ordinal})")


if __name__ == "__main__":
    main()
