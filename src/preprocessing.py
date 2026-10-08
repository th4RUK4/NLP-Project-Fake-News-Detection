"""Reusable text preprocessing utilities for the fake-news project."""

from __future__ import annotations

import re
import string
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _english_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return set()


_STOP_WORDS = _english_stopwords()
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: object) -> str:
    """Lowercase text and remove URLs, digits and punctuation."""
    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"\d+", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def remove_stopwords(text: str, words: Iterable[str] | None = None) -> str:
    """Remove English stopwords."""
    stop_set = _STOP_WORDS if words is None else set(words)
    return " ".join(word for word in text.split() if word not in stop_set)


def lemmatize_text(text: str) -> str:
    """Lemmatize words when the NLTK WordNet corpus is available."""
    try:
        return " ".join(_LEMMATIZER.lemmatize(word) for word in text.split())
    except LookupError:
        return text


def preprocess_text(text: object) -> str:
    """Run the project's deterministic preprocessing pipeline."""
    value = clean_text(text)
    value = remove_stopwords(value)
    return lemmatize_text(value)
