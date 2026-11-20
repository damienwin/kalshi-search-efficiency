"""Write the LLM prompts used in the search, exactly as logged, for the paper appendix.

Usage: python scripts/paper_appendix_prompts.py

Reads runs/llm_full/seed0/transcript.jsonl and runs/llm_no_history/seed0/transcript.jsonl
and writes plain-text files under paper/appendix/:

  prompt_system.txt          system prompt (byte-identical in every mode; checked here)
  prompt_user_full.txt       first-attempt user message of trial index 3 ("Trial 4 of 100"), LLM full
  prompt_user_nohistory.txt  the same trial, LLM no-history

Lines are wrapped at 96 characters (continuation lines indented) so they fit a
verbatim block, and the few non-ASCII characters are replaced with ASCII
equivalents for LaTeX. Nothing else is changed.
"""

import json
import os
import textwrap

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO_ROOT, "paper", "appendix")
WIDTH = 96
TRIAL = 3
ASCII = {"…": "...", "×": "x", "–": "-", "—": "-", "−": "-",
         "‘": "'", "’": "'", "“": '"', "”": '"', "≤": "<=", "≥": ">="}


def first_attempt(path: str, trial: int) -> dict:
    with open(path) as f:
        for line in f:
            e = json.loads(line)
            if e["trial"] == trial and e["attempt"] == 0:
                return e["request"]
    raise ValueError(f"no first attempt of trial {trial} in {path}")


def clean(text: str) -> str:
    for k, v in ASCII.items():
        text = text.replace(k, v)
    bad = sorted({c for c in text if ord(c) > 127})
    if bad:
        raise ValueError(f"unmapped non-ASCII characters: {bad}")
    out = []
    for line in text.split("\n"):
        indent = len(line) - len(line.lstrip(" "))
        out.extend(textwrap.wrap(line, WIDTH, subsequent_indent=" " * (indent + 4),
                                 break_long_words=True, break_on_hyphens=False) or [""])
    return "\n".join(out) + "\n"


def main():
    full = first_attempt(os.path.join(REPO_ROOT, "runs", "llm_full", "seed0", "transcript.jsonl"), TRIAL)
    nohist = first_attempt(os.path.join(REPO_ROOT, "runs", "llm_no_history", "seed0", "transcript.jsonl"), TRIAL)
    if full["system"] != nohist["system"]:
        raise ValueError("system prompts differ between modes")
    os.makedirs(OUT, exist_ok=True)
    files = {"prompt_system.txt": full["system"],
             "prompt_user_full.txt": full["messages"][0]["content"],
             "prompt_user_nohistory.txt": nohist["messages"][0]["content"]}
    for name, text in files.items():
        with open(os.path.join(OUT, name), "w") as f:
            f.write(clean(text))
        print(name, len(text), "chars")


if __name__ == "__main__":
    main()
