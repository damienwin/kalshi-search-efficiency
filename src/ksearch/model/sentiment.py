"""Frozen transformer sentiment scorers (ported from Spring src/model/sentiment.py).

Same models, truncation (512 tokens) and output shape as Spring:
  {label, positive, negative, neutral, confidence} per text.
"""

import logging
import time

log = logging.getLogger(__name__)

MODELS = {
    "finbert": "ProsusAI/finbert",
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}


class SentimentScorer:
    def __init__(self, name: str = "finbert", max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name, self.max_length, self.batch_size = name, max_length, batch_size
        start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS[name])
        self.model = AutoModelForSequenceClassification.from_pretrained(MODELS[name]).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        self.labels = [self.model.config.id2label[i].lower() for i in range(self.model.config.num_labels)]
        self._torch = torch
        log.info("loaded %s on %s in %.1fs", MODELS[name], self.device, time.time() - start)

    def score_texts(self, texts: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            for row in probs:
                s = {lab: float(p) for lab, p in zip(self.labels, row)}
                out.append({"label": max(s, key=s.get), "positive": s.get("positive", 0.0),
                            "negative": s.get("negative", 0.0), "neutral": s.get("neutral", 0.0),
                            "confidence": max(s.values())})
        return out
