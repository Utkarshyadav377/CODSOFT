"""
Movie Genre Classification Pipeline
====================================
Predicts movie genre from plot summaries using TF-IDF + multiple classifiers.
Dataset: ~54k training movies, ~54k test movies, 27 genres.
"""

import re
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_train(path):
    records = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                _, title, genre, description = parts
                records.append({"title": title.strip(),
                                 "genre": genre.strip().lower(),
                                 "description": description.strip()})
    return pd.DataFrame(records)


def load_test(path):
    records = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 3:
                _, title, description = parts
                records.append({"title": title.strip(),
                                 "description": description.strip()})
    return pd.DataFrame(records)


def load_solution(path):
    records = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                _, title, genre, description = parts
                records.append({"genre": genre.strip().lower()})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_features(df):
    """Combine title + description into one text field."""
    return (df["title"].fillna("") + " " + df["description"].fillna("")).apply(clean_text)


# ─────────────────────────────────────────────
# 3. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

def plot_genre_distribution(df, save_path="outputs/genre_distribution.png"):
    counts = df["genre"].value_counts()
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab20.colors
    bars = ax.bar(counts.index, counts.values, color=[colors[i % 20] for i in range(len(counts))])
    ax.set_title("Movie Genre Distribution (Training Set)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("Number of Movies", fontsize=12)
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=9)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(val), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_desc_length(df, save_path="outputs/description_length.png"):
    df = df.copy()
    df["desc_len"] = df["description"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["desc_len"].clip(0, 500), bins=60, color="#4C72B0", edgecolor="white")
    ax.set_title("Distribution of Plot Summary Word Counts", fontsize=14, fontweight="bold")
    ax.set_xlabel("Word Count", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    median = df["desc_len"].median()
    ax.axvline(median, color="red", linestyle="--", label=f"Median: {int(median)} words")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────

def make_pipelines():
    """Return dict of name → sklearn Pipeline."""
    tfidf_shared = dict(
        max_features=100_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_shared)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_shared)),
            ("clf",   LogisticRegression(
                max_iter=1000, C=5, solver="saga", n_jobs=-1)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_shared)),
            ("clf",   LinearSVC(C=1.0, max_iter=2000)),
        ]),
    }


def train_and_evaluate(train_df, test_df, test_labels):
    X_train = build_features(train_df)
    y_train = train_df["genre"]
    X_test  = build_features(test_df)
    y_test  = test_labels["genre"]

    results = {}
    pipelines = make_pipelines()

    for name, pipe in pipelines.items():
        print(f"\n  ▶ Training {name}...")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = pipe.predict(X_test)
        infer_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        macro_f1 = report["macro avg"]["f1-score"]

        results[name] = {
            "pipeline": pipe,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "train_time": train_time,
            "infer_time": infer_time,
            "y_pred": y_pred,
            "report": report,
        }
        print(f"    Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f} | "
              f"Train: {train_time:.1f}s | Infer: {infer_time:.2f}s")

    return results


# ─────────────────────────────────────────────
# 5. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_model_comparison(results, save_path="outputs/model_comparison.png"):
    names  = list(results.keys())
    acc    = [results[n]["accuracy"] for n in names]
    f1     = [results[n]["macro_f1"] for n in names]
    times  = [results[n]["train_time"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    palette = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, vals, title, fmt in zip(
        axes,
        [acc, f1, times],
        ["Test Accuracy", "Macro-F1 Score", "Training Time (s)"],
        [".4f", ".4f", ".1f"]
    ):
        bars = ax.bar(names, vals, color=palette, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.15)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f"{val:{fmt}}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticklabels(names, rotation=10, ha="right")
        ax.spines[["top","right"]].set_visible(False)

    fig.suptitle("Model Comparison — Movie Genre Classification", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(results, y_test, top_n=10, save_path="outputs/confusion_matrix.png"):
    """Plot confusion matrix for best model on top-N genres."""
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best = results[best_name]

    # Restrict to top genres for readability
    genre_counts = pd.Series(y_test).value_counts()
    top_genres   = genre_counts.head(top_n).index.tolist()
    mask = pd.Series(y_test).isin(top_genres)

    y_true_top = pd.Series(y_test)[mask.values]
    y_pred_top = pd.Series(best["y_pred"])[mask.values]

    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_genres)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=top_genres, yticklabels=top_genres,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Normalised Recall"})
    ax.set_title(f"Confusion Matrix — {best_name} (Top {top_n} Genres)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Genre", fontsize=12)
    ax.set_ylabel("True Genre", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_genre_f1(results, save_path="outputs/per_genre_f1.png"):
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    report    = results[best_name]["report"]

    genres = [g for g in report if g not in ("accuracy", "macro avg", "weighted avg")]
    f1s    = [report[g]["f1-score"] for g in genres]
    supports = [report[g]["support"] for g in genres]

    sorted_pairs = sorted(zip(genres, f1s, supports), key=lambda x: -x[1])
    genres_s, f1s_s, sup_s = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#2ecc71" if f >= 0.6 else "#e67e22" if f >= 0.35 else "#e74c3c" for f in f1s_s]
    bars = ax.bar(genres_s, f1s_s, color=colors, edgecolor="white")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.6, color="green", linestyle="--", alpha=0.5, label="F1 = 0.60")
    ax.axhline(0.35, color="orange", linestyle="--", alpha=0.5, label="F1 = 0.35")
    ax.set_title(f"Per-Genre F1 Score — {best_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Genre", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_xticklabels(genres_s, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    for bar, val in zip(bars, f1s_s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_top_tfidf_features(results, top_n=15, save_path="outputs/top_features.png"):
    """Show top TF-IDF features for key genres (Logistic Regression)."""
    name = "Logistic Regression"
    if name not in results:
        return

    pipe = results[name]["pipeline"]
    tfidf = pipe.named_steps["tfidf"]
    clf   = pipe.named_steps["clf"]
    classes = clf.classes_
    feature_names = np.array(tfidf.get_feature_names_out())

    target_genres = ["drama", "comedy", "thriller", "horror", "documentary", "action"]
    target_genres = [g for g in target_genres if g in classes]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, genre in zip(axes, target_genres):
        idx = list(classes).index(genre)
        coefs = clf.coef_[idx]
        top_idx = np.argsort(coefs)[-top_n:][::-1]
        top_words = feature_names[top_idx]
        top_vals  = coefs[top_idx]

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))
        ax.barh(range(top_n), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_words[::-1], fontsize=9)
        ax.set_title(f"Genre: {genre.upper()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("LR Coefficient", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)

    for ax in axes[len(target_genres):]:
        ax.set_visible(False)

    fig.suptitle(f"Top {top_n} Discriminative Words per Genre (Logistic Regression)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
# 6. INFERENCE HELPER
# ─────────────────────────────────────────────

def predict_genre(text, pipeline, top_k=3):
    """Return top-k predicted genres with probabilities (if available)."""
    clean = clean_text(text)
    clf = pipeline.named_steps["clf"]
    tfidf = pipeline.named_steps["tfidf"]
    X = tfidf.transform([clean])

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        top_idx = np.argsort(probs)[-top_k:][::-1]
        return [(clf.classes_[i], probs[i]) for i in top_idx]
    else:
        pred = clf.predict(X)[0]
        return [(pred, 1.0)]


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    import os
    os.makedirs("outputs", exist_ok=True)

    print("=" * 60)
    print("  MOVIE GENRE CLASSIFICATION PIPELINE")
    print("=" * 60)

    # --- Load data ---
    print("\n[1/5] Loading data...")
    train_df  = load_train("data/train_data.txt")
    test_df   = load_test("data/test_data.txt")
    test_sol  = load_solution("data/test_data_solution.txt")

    print(f"  Train: {len(train_df):,} samples | Genres: {train_df['genre'].nunique()}")
    print(f"  Test : {len(test_df):,}  samples")
    print(f"  Genres: {sorted(train_df['genre'].unique())}")

    # --- EDA plots ---
    print("\n[2/5] Generating EDA plots...")
    plot_genre_distribution(train_df)
    plot_desc_length(train_df)

    # --- Training & Evaluation ---
    print("\n[3/5] Training models...")
    results = train_and_evaluate(train_df, test_df, test_sol)

    # --- Comparison plots ---
    print("\n[4/5] Generating evaluation plots...")
    plot_model_comparison(results)
    y_test = test_sol["genre"].values
    plot_confusion_matrix(results, y_test)
    plot_per_genre_f1(results)
    plot_top_tfidf_features(results)

    # --- Summary table ---
    print("\n[5/5] Final Summary")
    print("-" * 60)
    print(f"{'Model':<22} {'Accuracy':>10} {'Macro-F1':>10} {'Train(s)':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<22} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} {r['train_time']:>10.1f}")
    print("-" * 60)

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_pipe = results[best_name]["pipeline"]
    print(f"\n  Best model: {best_name}")

    # --- Save best model ---
    joblib.dump(best_pipe, "outputs/best_model.joblib")
    print("  Model saved to outputs/best_model.joblib")

    # --- Demo predictions ---
    print("\n─── Demo Predictions ───")
    demos = [
        "A detective investigates a series of brutal murders in a rain-soaked city.",
        "Two unlikely friends embark on a hilarious road trip across America.",
        "Scientists discover a new galaxy and must travel light years to explore it.",
        "A family moves into a haunted house and strange events begin to occur.",
        "A biopic about a jazz musician rising to fame in 1950s New Orleans.",
    ]
    for text in demos:
        preds = predict_genre(text, best_pipe, top_k=3)
        top_genre = preds[0][0]
        print(f"\n  Plot: \"{text[:70]}...\"")
        print(f"  → Predicted: {top_genre.upper()}  |  Top-3: {[(g, f'{p:.2f}') for g,p in preds]}")

    print("\n✓ Done. All outputs saved to outputs/")


if __name__ == "__main__":
    main()
