"""Topic relevance filter: terms come from the title, matching reads headline + lead only."""

from ksearch.data.relevance import filter_relevant, is_relevant, topic_terms


def test_terms_drop_contract_words_dates_and_numbers():
    assert topic_terms("Will the high temp in Chicago be 75-76° on Mar 28, 2025?") == ["chicago"]
    assert topic_terms("Bitcoin price on Jun 7, 2026?") == ["bitcoin"]
    assert topic_terms("Will the WTI crude oil settlement price be above 93.99 USD/Bbl on Sep 9, 2026?") == [
        "wti", "crude", "oil"]


def test_sp500_keeps_its_name():
    assert topic_terms("Will the S&P 500 be above 5199.99 at the end of May 13, 2024?") == ["sp"]


def test_terms_are_capped_distinct_and_ordered():
    t = topic_terms("Alpha beta gamma delta epsilon alpha")
    assert t == ["alpha", "beta", "gamma", "delta"]


def test_matches_headline_or_lead_whole_words_only():
    assert is_relevant(["chicago"], "Chicago braces for record cold")
    assert is_relevant(["chicago"], "Cold snap", lead="Temperatures in Chicago fell")
    assert not is_relevant(["chicago"], "Chicagoland retailers expand")
    assert not is_relevant(["chicago"], "Spin a yarn: why the knitted dress is an easy look")


def test_body_text_is_never_read():
    assert not is_relevant(["bitcoin"], "Spin a yarn", lead="x" * 400 + " bitcoin")


def test_aliases_match_related_headlines():
    assert is_relevant(["sp"], "Wall Street ends higher as tech rebounds")
    assert is_relevant(["wti", "crude"], "Oil jumps after OPEC cut")
    assert not is_relevant(["sp"], "Spinning yarn")


def test_no_terms_means_nothing_is_relevant():
    assert not is_relevant([], "Anything at all")


def test_filter_relevant_keeps_order():
    arts = [{"title": "Bitcoin hits high"}, {"title": "Knitting"}, {"title": "Crypto slump", "lead": "bitcoin down"}]
    assert [a["title"] for a in filter_relevant(["bitcoin"], arts)] == ["Bitcoin hits high", "Crypto slump"]
