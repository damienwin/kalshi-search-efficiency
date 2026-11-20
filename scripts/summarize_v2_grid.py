import glob, json, re, statistics as st
R = "/Users/damienwin/harness-projects/kalshi-search-efficiency/results/"
CONFIGS = [("price", "cv_price_v2"), ("price, news-only", "cv_newsonly_price_v2"),
           ("price+sent vs price", "cv_sentiment_vs_price_v2"),
           ("price+sent vs price, news-only", "cv_sentiment_vs_price_newsonly_v2"),
           ("sent-only, news-only", "cv_newsonly_sentiment_v2")]
ms = lambda xs: f"{st.mean(xs):.4f} ± {st.stdev(xs):.4f}"
for scope in ["all", "spring", "current"]:
    print(f"\n=== scope: {scope} ===")
    for name, stem in CONFIGS:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{R}{stem}_{scope}_seed[0-4].json"))]
        if not runs: continue
        r0 = runs[0]
        print(f"\n[{name}]  n={r0['n_events_scored']}  seeds={len(runs)}  labels={r0['label_dist_scored']}")
        for k in r0["pooled_macro_f1"]:
            print(f"  F1 {k:<10} {ms([r['pooled_macro_f1'][k] for r in runs])}")
        for c in r0["comparisons"]:
            d = [r["comparisons"][c]["delta"] for r in runs]
            lo = [r["comparisons"][c]["ci95"][0] for r in runs]; hi = [r["comparisons"][c]["ci95"][1] for r in runs]
            sig = sum(r["comparisons"][c]["excludes_zero"] for r in runs)
            print(f"  {c:<22} Δ {ms(d)}  [{min(d):+.4f},{max(d):+.4f}]  pos {sum(x>0 for x in d)}/{len(d)}"
                  f"  CI(median) [{st.median(lo):+.4f},{st.median(hi):+.4f}]  sig {sig}/{len(d)}")
