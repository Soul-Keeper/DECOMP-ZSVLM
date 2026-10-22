"""
Check of the label parser against a human reading (Section V-D, Appendix A).

    python -m src.parser_check score              # agreement, from the released reading
    python -m src.parser_check sample             # redraw the sample from the predictions
    python -m src.parser_check read --answers my_reading.json
    python -m src.parser_check score --answers my_reading.json

metadata/parser_check.json holds the 200 sampled responses, the stratum of
each, the parser outcome and the reading recorded by one author. The reading
was blind: the reader saw only the question and the response.

sample draws again from the stored classification responses of all models
except CLIP, single-task runs only, and checks that it obtains the released
sample. Responses that consist of a bare label name are counted but not
sampled; the rest are split by task and parser outcome (label extracted or
response rejected), and 50 are drawn from each of the four strata.

read runs the blind reading in the terminal, so that it can be repeated by
another reader; score then compares that reading with the parser.
"""

import argparse
import json
import os
import re
import shutil
import sys
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np

from src.loading import discover
from src.parsing import COLOR_LABELS, TYPE_LABELS, is_contrastive, parse_color, parse_type

LABELS = {"type": TYPE_LABELS, "color": COLOR_LABELS}
PARSE = {"type": parse_type, "color": parse_color}
STRATA = ["type/extracted", "type/rejected", "color/extracted", "color/rejected"]
SEED = 20261020
PER_STRATUM = 50

RULES = """
Record the answer the model commits to, in its own words. Correctness does not matter.

  digit  the model names this label (case, hyphen, space and plural do not matter)
  word   the model names a single term outside the list: type it as written
  0      no single answer

  1. A final answer overrides the reasoning before it.
  2. Hedging, or listing options without choosing one, is no answer.
  3. When several objects are described, take the one the question asks about
     only if this is unambiguous.
  4. A single term outside the list is recorded as written.
  5. Refusals, irrelevant, empty or truncated responses are no answer.

  b  back to the previous response    ?  these rules    q  quit (progress is saved)
"""


def stratum(task, raw):
    text = re.sub(r'[\s.!*"]+$', "", (raw or "").strip().lower()).strip('*" ')
    if text in LABELS[task]:
        return "trivial"
    return "extracted" if PARSE[task](raw) else "rejected"


# ── sample ─────────────────────────────────────────────────────────────────

def sample(args):
    pool, population = {s: [] for s in STRATA}, Counter()
    for (model, strategy), path in sorted(discover(Path(args.predictions) / "main_benchmark").items()):
        if strategy == "descriptive_multitask" or is_contrastive(model):
            continue
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        for task in ["type", "color"]:
            for r in rows:
                kind = stratum(task, r.get(f"raw_{task}"))
                population[f"{task}/{kind}"] += 1
                if kind != "trivial":
                    pool[f"{task}/{kind}"].append((model, strategy, r["file_name"], task))
    rng = np.random.default_rng(SEED)
    items = []
    for s in STRATA:
        items += [pool[s][i] for i in rng.choice(len(pool[s]), PER_STRATUM, replace=False)]
    items = [items[i] for i in rng.permutation(len(items))]

    total = sum(population.values())
    trivial = population["type/trivial"] + population["color/trivial"]
    print(f"classification responses: {total}, bare label names: {trivial} ({trivial / total:.1%})")
    for s in STRATA:
        print(f"  {s:<16}{population[s]}")
    released = json.load(open(Path(args.metadata) / "parser_check.json", encoding="utf-8"))
    same = [(it["model"], it["strategy"], it["file_name"], it["task"]) for it in released["items"]] == items
    print(f"sample identical to metadata/parser_check.json: {same}")


# ── read ───────────────────────────────────────────────────────────────────

def read(args):
    items = json.load(open(Path(args.metadata) / "parser_check.json", encoding="utf-8"))["items"]
    path = Path(args.answers)
    answers = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    width = min(shutil.get_terminal_size((100, 20)).columns, 110)
    print(RULES)
    i = 0
    while i < len(items):
        item = items[i]
        if item["id"] in answers and not args.review:
            i += 1
            continue
        labels = LABELS[item["task"]]
        print("\n" + "=" * width)
        print(f'[{i + 1}/{len(items)}]  {item["id"]}   {item["task"].upper()}')
        print("\nQUESTION:")
        for line in item["question"].replace("<image>", "").strip().splitlines():
            print(textwrap.fill(line, width, initial_indent="  ", subsequent_indent="  "))
        print("\nRESPONSE:")
        for line in (item["response"] if item["response"].strip() else "(empty)").splitlines():
            print(textwrap.fill(line, width, initial_indent="  ", subsequent_indent="  ") if line else "")
        if item["id"] in answers:
            print(f'  (recorded: {answers[item["id"]]["value"] or "none"})')
        print("\n  " + "   ".join(f"{k}={label}" for k, label in enumerate(labels, 1)) + "   0=none   or a word")
        try:
            raw = input("> ").strip()
        except EOFError:
            break
        low = raw.lower()
        if low == "q":
            break
        if low == "?":
            print(RULES)
            continue
        if low == "b":
            i = max(i - 1, 0)
            args.review = True
            continue
        if low.isdigit() and int(low) <= len(labels):
            answer = {"kind": "label", "value": labels[int(low) - 1]} if int(low) else {"kind": "none", "value": None}
        elif low and not low.isdigit():
            answer = {"kind": "word", "value": raw}
        else:
            continue
        answers[item["id"]] = answer
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(answers, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        args.review = False
        i += 1
    print(f"\n{len(answers)} of {len(items)} read, saved to {path}")


# ── score ──────────────────────────────────────────────────────────────────

def score(args):
    released = json.load(open(Path(args.metadata) / "parser_check.json", encoding="utf-8"))
    items, population = released["items"], released["population"]
    reading = ({it["id"]: it["reading"] for it in items} if args.answers is None
               else json.load(open(args.answers, encoding="utf-8")))
    missing = [it["id"] for it in items if it["id"] not in reading]
    if missing:
        sys.exit(f"{len(missing)} responses not read, e.g. {missing[0]}")

    rows = []
    for it in items:
        value = reading[it["id"]]["value"]
        reader = PARSE[it["task"]](value) if value else None
        kind = "none" if value is None else ("label" if reader else "outside")
        rows.append(dict(it, reader=reader, kind=kind))

    print(f'{"stratum":<17}{"agree":>8}{"population":>12}')
    agreed, weight = 0.0, 0
    for s in STRATA:
        group = [r for r in rows if r["stratum"] == s]
        k = sum(r["reader"] == r["parser_label"] for r in group)
        agreed += population[s] * k / len(group)
        weight += population[s]
        print(f"{s:<17}{f'{k}/{len(group)}':>8}{population[s]:>12}")
    trivial = population["type/trivial"] + population["color/trivial"]
    total = weight + trivial
    print(f"bare label names: {trivial} of {total} ({trivial / total:.1%}), parsed as written")
    print(f"agreement, weighted by stratum size: {agreed / weight:.1%} of responses that are not a bare label, "
          f"{(agreed + trivial) / total:.1%} of all responses")

    wrong = [r for r in rows if r["reader"] != r["parser_label"]]
    print(f"\ndisagreements: {len(wrong)}, by strategy {dict(Counter(r['strategy'] for r in wrong))}")
    print(f"  reader found a label the parser rejected: {sum(r['parser_label'] is None for r in wrong)}")
    print(f"  parser accepted a label the reader did not: {sum(r['parser_label'] is not None for r in wrong)}")
    rejected = [r for r in rows if r["stratum"].endswith("rejected")]
    kinds = Counter(r["kind"] for r in rejected)
    print(f"\nrejected responses: {len(rejected)}; a term outside the label set {kinds['outside']}, "
          f"no single answer {kinds['none']}, a label the reader committed to {kinds['label']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["sample", "read", "score"]:
        p = sub.add_parser(name)
        p.add_argument("--metadata", default="metadata")
        if name == "sample":
            p.add_argument("--predictions", default="predictions")
        if name == "read":
            p.add_argument("--answers", default="parser_check_answers.json")
            p.add_argument("--review", action="store_true", help="go through responses already read")
        if name == "score":
            p.add_argument("--answers", default=None, help="a reading made with read; default: the released one")
    args = ap.parse_args()
    {"sample": sample, "read": read, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
