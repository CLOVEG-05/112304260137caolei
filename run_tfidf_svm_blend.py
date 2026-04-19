import argparse
import csv
import html
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT.parent / "word2vec-nlp-tutorial"
CACHE_DIR = ROOT / "artifacts" / "cache"
REPORTS_DIR = ROOT / "reports"
SUBMISSIONS_DIR = ROOT / "submissions"

CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mustn't": "must not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
}

NEGATION_WORDS = {"no", "nor", "not", "never"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the 0.96+ TF-IDF + SVM blend experiment.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing labeledTrainData.tsv and testData.tsv",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore the local preprocessed cache and rebuild it.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    for path in [CACHE_DIR, REPORTS_DIR, SUBMISSIONS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_pickle(path: Path, payload) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_competition_tsv(path: Path) -> pd.DataFrame:
    kwargs = {"sep": "\t"}
    if path.name == "unlabeledTrainData.tsv":
        kwargs["quoting"] = csv.QUOTE_NONE
    return pd.read_csv(path, **kwargs)


def normalize_review(review: str) -> str:
    if pd.isna(review):
        return ""

    text = html.unescape(str(review))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()

    for short, long_form in CONTRACTIONS.items():
        text = text.replace(short, long_form)

    text = re.sub(r"[^a-z!?'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def soft_clean(review: str) -> str:
    text = html.unescape(str(review))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_review(clean_review: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(clean_review)
    return [token for token in tokens if len(token) > 1 and token not in STOP_WORDS]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["clean_review"] = prepared["review"].map(normalize_review)
    prepared["tokens"] = prepared["clean_review"].map(tokenize_review)
    prepared["joined_tokens"] = prepared["tokens"].map(" ".join)
    prepared["soft_review"] = prepared["review"].map(soft_clean)
    return prepared


def load_or_preprocess(data_dir: Path, rebuild_cache: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_cache = CACHE_DIR / "labeled_tfidf_svm.pkl"
    test_cache = CACHE_DIR / "test_tfidf_svm.pkl"

    if not rebuild_cache and labeled_cache.exists() and test_cache.exists():
        return load_pickle(labeled_cache), load_pickle(test_cache)

    labeled = read_competition_tsv(data_dir / "labeledTrainData.tsv")
    test = read_competition_tsv(data_dir / "testData.tsv")

    labeled = preprocess_dataframe(labeled)
    test = preprocess_dataframe(test)

    save_pickle(labeled_cache, labeled)
    save_pickle(test_cache, test)
    return labeled, test


def rank_normalize(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True).to_numpy()


def main() -> None:
    args = parse_args()
    ensure_directories()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    labeled, test = load_or_preprocess(data_dir, rebuild_cache=args.rebuild_cache)

    train_df, val_df = train_test_split(
        labeled,
        test_size=0.2,
        random_state=42,
        stratify=labeled["sentiment"],
    )

    y_train = train_df["sentiment"].astype(int).to_numpy()
    y_val = val_df["sentiment"].astype(int).to_numpy()
    y_full = labeled["sentiment"].astype(int).to_numpy()

    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=200000,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.99,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=200000,
    )

    x_train_word = word_vectorizer.fit_transform(train_df["joined_tokens"])
    x_val_word = word_vectorizer.transform(val_df["joined_tokens"])
    x_full_word = word_vectorizer.fit_transform(labeled["joined_tokens"])
    x_test_word = word_vectorizer.transform(test["joined_tokens"])

    x_train_char = char_vectorizer.fit_transform(train_df["soft_review"])
    x_val_char = char_vectorizer.transform(val_df["soft_review"])
    x_full_char = char_vectorizer.fit_transform(labeled["soft_review"])
    x_test_char = char_vectorizer.transform(test["soft_review"])

    word_model = LinearSVC(C=0.6)
    char_model = LinearSVC(C=0.4)

    word_model.fit(x_train_word, y_train)
    char_model.fit(x_train_char, y_train)

    val_word_score = word_model.decision_function(x_val_word)
    val_char_score = char_model.decision_function(x_val_char)

    val_word_rank = rank_normalize(val_word_score)
    val_char_rank = rank_normalize(val_char_score)
    validation_blend = 0.8 * val_word_rank + 0.2 * val_char_rank

    validation_auc = float(roc_auc_score(y_val, validation_blend))
    validation_word_auc = float(roc_auc_score(y_val, val_word_rank))
    validation_char_auc = float(roc_auc_score(y_val, val_char_rank))

    word_model.fit(x_full_word, y_full)
    char_model.fit(x_full_char, y_full)

    test_word_score = word_model.decision_function(x_test_word)
    test_char_score = char_model.decision_function(x_test_char)
    test_blend = 0.8 * rank_normalize(test_word_score) + 0.2 * rank_normalize(test_char_score)

    submission = pd.DataFrame({"id": test["id"], "sentiment": test_blend})
    submission_path = SUBMISSIONS_DIR / "submission_tfidf_svm_blend_auc.csv"
    submission.to_csv(submission_path, index=False)

    report = {
        "metric_priority": "roc_auc",
        "model": "word_tfidf_linear_svc + char_tfidf_linear_svc",
        "weights": {"word_svc": 0.8, "char_svc": 0.2},
        "word_config": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "max_features": 200000,
            "C": 0.6,
        },
        "char_config": {
            "ngram_range": [3, 5],
            "analyzer": "char_wb",
            "min_df": 2,
            "max_df": 0.99,
            "max_features": 200000,
            "C": 0.4,
        },
        "component_validation_auc": {
            "word_svc": validation_word_auc,
            "char_svc": validation_char_auc,
        },
        "validation_auc": validation_auc,
        "submission_path": str(submission_path),
    }

    report_path = REPORTS_DIR / "tfidf_svm_blend_validation_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Validation ROC-AUC: {validation_auc:.8f}")
    print(f"Saved submission to: {submission_path}")
    print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
