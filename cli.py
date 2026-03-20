"""
CLI for the SPO Relationship Extractor.

Usage examples:
  # From a text string
  python cli.py --text "Apple was founded by Steve Jobs in 1976."

  # From a file
  python cli.py --file report.txt

  # Include implicit relationships
  python cli.py --file report.txt --implicit

  # Choose output format (table | csv | json | triplets)
  python cli.py --file report.txt --format json
  python cli.py --file report.txt --format csv --output results.csv

  # Use a different model
  python cli.py --text "..." --model gpt-4o
"""

import argparse
import io
import json
import os
import sys

from dotenv import load_dotenv, find_dotenv


def _load_text_from_file(path: str) -> str:
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    if path.lower().endswith(".pdf"):
        try:
            import PyPDF2
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print("Error: PyPDF2 not installed. Run: pip install PyPDF2", file=sys.stderr)
            sys.exit(1)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def _render_table(rows: list[dict]) -> str:
    if not rows:
        return "(no relationships found)"
    cols = ["subject", "predicate", "object", "negated"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    sep = "+" + "+".join("-" * (widths[c] + 2) for c in cols) + "+"
    header = "|" + "|".join(f" {c:<{widths[c]}} " for c in cols) + "|"
    lines = [sep, header, sep]
    for r in rows:
        line = "|" + "|".join(f" {str(r.get(c,'')):<{widths[c]}} " for c in cols) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def _render_csv(rows: list[dict]) -> str:
    cols = ["subject", "predicate", "object", "negated", "source_quote"]
    out = io.StringIO()
    out.write(",".join(cols) + "\n")
    for r in rows:
        out.write(",".join(f'"{str(r.get(c,"")).replace(chr(34), chr(34)*2)}"' for c in cols) + "\n")
    return out.getvalue()


def _render_triplets(rows: list[dict]) -> str:
    return "\n".join(
        f"{r['subject']} | {r['predicate']} | {r['object']}"
        for r in rows
    )


def main():
    load_dotenv(find_dotenv(), override=True)

    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="Extract subject-predicate-object relationships from text.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", "-t", help="Input text directly as a string")
    input_group.add_argument("--file", "-f", help="Path to a .txt, .md, or .pdf file")

    parser.add_argument(
        "--format", choices=["table", "csv", "json", "triplets"],
        default="table", help="Output format (default: table)"
    )
    parser.add_argument("--output", "-o", help="Write output to this file instead of stdout")
    parser.add_argument("--implicit", action="store_true", help="Also extract implicit relationships (Pass 3)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--api-key", help="OpenAI API key (overrides OPENAI_API_KEY env var)")

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Use --api-key or add it to .env", file=sys.stderr)
        sys.exit(1)

    # Load text
    text = args.text if args.text else _load_text_from_file(args.file)
    if not text.strip():
        print("Error: input text is empty.", file=sys.stderr)
        sys.exit(1)

    # Extract
    from relationship_extractor import RelationshipExtractor
    extractor = RelationshipExtractor(api_key=api_key, model=args.model)

    print(f"Extracting relationships (model={args.model}, implicit={args.implicit})...", file=sys.stderr)
    df = extractor.extract(text, include_implicit=args.implicit)
    rows = df.to_dict(orient="records")

    # Format output
    if args.format == "table":
        output = _render_table(rows)
        output += f"\n\n{len(rows)} relationship(s) found."
    elif args.format == "csv":
        output = _render_csv(rows)
    elif args.format == "json":
        output = json.dumps(rows, indent=2)
    else:  # triplets
        output = _render_triplets(rows)

    # Write
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Results written to {args.output} ({len(rows)} relationships)", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
