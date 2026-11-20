"""Topic relevance filter for matched news articles.

The Guardian query is built from overlapping word pairs of the market title, so
most returned articles never mention the market's subject (Week 3 report: 3.4%
contain the query phrase). This module derives a few topic terms from the title
and keeps an article only if its headline or lead mentions one of them.

Pure functions, no I/O. Matching reads the headline and lead only: body text
mentions anything, which is how unrelated articles got matched in the first place.
"""

import re

# Words that describe the contract, not its subject.
GENERIC = {
    "price", "prices", "above", "below", "between", "end", "open", "close", "closes", "closing",
    "high", "low", "temp", "range", "settlement", "settle", "usd", "bbl", "than", "over", "under",
    "will", "year", "month", "week", "day", "daily", "weekly", "monthly", "average", "avg",
    "level", "value", "rate", "percent", "pct", "before", "after", "least", "most", "new",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun", "edt", "est", "pst", "pdt", "utc", "am", "pm",
    "yes", "not", "any", "total", "number", "per",
}
STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "are", "was", "were", "has", "have",
    "its", "than", "who", "what", "when", "where", "how", "which", "does", "did", "can", "could",
    "would", "should",
}
# Subject aliases: a term also matches these words in a headline.
ALIASES = {
    "sp": ["s&p", "wall street", "stocks", "us stocks", "stock market"],
    "nasdaq": ["nasdaq", "tech stocks", "wall street"],
    "bitcoin": ["bitcoin", "crypto"],
    "btc": ["bitcoin", "crypto"],
    "ethereum": ["ethereum", "crypto", "ether"],
    "eth": ["ethereum", "crypto", "ether"],
    "wti": ["oil", "crude", "opec"],
    "crude": ["oil", "crude", "opec"],
    "usdjpy": ["yen", "japan"],
    "jpy": ["yen", "japan"],
    "cpi": ["inflation", "consumer prices"],
    "payrolls": ["jobs report", "payrolls", "employment", "unemployment"],
    "gas": ["gas prices", "petrol", "fuel"],
}

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9&]*")


def topic_terms(title: str) -> list[str]:
    """Distinct lowercase subject terms from a market title (at most 4), in title order."""
    text = (title or "").replace("S&P", "SP").replace("s&p", "sp")
    seen, out = set(), []
    for tok in _TOKEN.findall(text):
        t = tok.lower()
        if t in GENERIC or t in STOP or (len(t) < 3 and t != "sp") or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:4]


def _patterns(terms: list[str]) -> list[re.Pattern]:
    words = []
    for t in terms:
        words.extend(ALIASES.get(t, [t]))
    return [re.compile(r"(?<![a-z0-9])" + re.escape(w.lower()) + r"(?![a-z0-9])") for w in dict.fromkeys(words)]


def lead_text(title: str, lead: str = "", max_lead_chars: int = 300) -> str:
    """Headline plus the first part of the lead, lowercased."""
    return f"{title or ''} {(lead or '')[:max_lead_chars]}".lower()


def is_relevant(terms: list[str], title: str, lead: str = "") -> bool:
    """True if the headline or lead mentions any topic term (or its alias)."""
    if not terms:
        return False
    text = lead_text(title, lead)
    return any(p.search(text) for p in _patterns(terms))


def filter_relevant(terms: list[str], articles: list[dict]) -> list[dict]:
    """Keep articles whose 'title' / 'lead' mention the topic; each article is a dict."""
    return [a for a in articles if is_relevant(terms, a.get("title", ""), a.get("lead", ""))]
