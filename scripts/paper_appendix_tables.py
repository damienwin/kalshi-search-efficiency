"""Write the appendix tables of per-run results and accepted features for the paper.

    python scripts/paper_appendix_tables.py

Reads runs/<arm>/seed<k>/{run.json,trials.jsonl} (dev search only) and writes
  paper/tables/per_run.tex        gain, accepted count and macro-F1 change of every run
  paper/tables/accepted.tex       every accepted feature, with its gain, DM p and placebo margin
"""

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO_ROOT, "runs")
OUT = os.path.join(REPO_ROOT, "paper", "tables")
ARMS = [("random", "Random"), ("tpe", "TPE"), ("llm_full", "LLM full"), ("llm_no_history", "LLM no-history")]
SEEDS = range(5)


def tex(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def per_run() -> str:
    gain_rows, f1_rows = [], []
    for arm, name in ARMS:
        g, f = [], []
        for s in SEEDS:
            r = json.load(open(os.path.join(RUNS, arm, f"seed{s}", "run.json")))
            ini, fin = r["initial_reference"], r["final_reference"]
            gain = ini["balanced_logloss"] - fin["balanced_logloss"]
            df1 = fin["pooled_macro_f1"] - ini["pooled_macro_f1"]
            n = r["totals"]["accepted"]
            g.append(f"{gain:.4f} ({n})")
            f.append(f"{df1:+.3f}".replace("-0.000", "+0.000"))
        gain_rows.append(f"    {name} & " + " & ".join(g) + r" \\")
        f1_rows.append(f"    {name} & " + " & ".join(f) + r" \\")
    head = " & ".join(f"\\textbf{{Seed {s}}}" for s in SEEDS)
    return "\n".join([
        r"\begin{table}[!ht]",
        r"  \caption{Every development run, by the seed of the search. Top: gain in class-balanced log loss from the base classifier to the run's final reference classifier, with the number of accepted features in parentheses. Bottom: change in pooled macro F1 over the same step.}",
        r"  \label{tab:per-run}",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lccccc}",
        r"    \toprule",
        f"    \\textbf{{Arm}} & {head} \\\\",
        r"    \midrule",
        r"    \multicolumn{6}{l}{\emph{Gain (accepted)}} \\",
        *gain_rows,
        r"    \midrule",
        r"    \multicolumn{6}{l}{\emph{$\Delta$ macro F1}} \\",
        *f1_rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ])


def accepted() -> str:
    rows = []
    for i, (arm, name) in enumerate(ARMS[:3]):
        if i:
            rows.append(r"    \midrule")
        for s in SEEDS:
            for line in open(os.path.join(RUNS, arm, f"seed{s}", "trials.jsonl")):
                t = json.loads(line)
                if t["status"] != "accepted":
                    continue
                g = t["gate"]
                margin = g["mean_loss_gain"] - max(t["placebo_gains"])
                rows.append(f"    {name} & {s} & {t['trial']} & \\texttt{{{tex(t['formula'])}}} & "
                            f"{g['mean_loss_gain']:.4f} & {g['dm_p']:.3f} & {margin:.4f} \\\\")
    return "\n".join([
        r"\begin{table}[!ht]",
        r"  \caption{All 26 features accepted on the development set (the no-history arm accepted none). Gain: reduction in class-balanced log loss against the reference classifier at that trial, which already includes the run's earlier acceptances. DM $p$: one-sided Diebold--Mariano $p$-value. Margin: gain minus the best of the 19 placebo copies. Operator definitions are in Table~\ref{tab:operators}.}",
        r"  \label{tab:accepted}",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \begin{tabular}{lcc>{\raggedright\arraybackslash}p{7.2cm}ccc}",
        r"    \toprule",
        r"    \textbf{Arm} & \textbf{Seed} & \textbf{Trial} & \textbf{Feature} & \textbf{Gain} & \textbf{DM $p$} & \textbf{Margin} \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]), len(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "per_run.tex"), "w") as f:
        f.write(per_run())
    body, n = accepted()
    assert n == 26 + 2, n  # 26 features + 2 separator rules
    with open(os.path.join(OUT, "accepted.tex"), "w") as f:
        f.write(body)
    print(f"wrote {OUT}/per_run.tex and accepted.tex ({n - 2} features)")
