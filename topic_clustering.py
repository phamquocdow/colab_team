"""Reproducible, label-free topic clustering for notebook 5.

Inputs are the English ``clean_text`` strings produced by notebook 3. Source
labels and subject metadata are deliberately absent from model selection.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


# Fixed vocabulary of conversational words, publishing boilerplate and broad
# forms of address. Do not derive this list from Fake/True labels or subjects.
# This filter is specific to topic discovery; classification stays unchanged.
STYLE_WORDS = frozenset("""
say said says tell told get got go going gone come came make made take took
want know think thing really also much many would could even one two people
like look well way back time year day today new first last video watch image
photo picture featured via subscribe share read click twitter facebook youtube
com www http https mr ms donald president
""".split())
TOPIC_STOP_WORDS = sorted(ENGLISH_STOP_WORDS | STYLE_WORDS)


def build_features(texts, topic_focused=True, seed=42, dimensions=100):
    """Fit TF-IDF and normalized LSA; return fitted stages and both matrices."""
    # Match the original notebook's float64 calculation. Precision changes can
    # alter randomized SVD / K-Means local optima and obscure the comparison.
    params = dict(sublinear_tf=True, dtype=np.float64)
    if topic_focused:
        params.update(max_features=30_000, min_df=10, max_df=0.4,
                      ngram_range=(1, 2), stop_words=TOPIC_STOP_WORDS)
    else:
        params.update(max_features=20_000, min_df=5, max_df=0.5)
    tfidf = TfidfVectorizer(**params)
    sparse = tfidf.fit_transform(texts)
    if min(sparse.shape) <= dimensions:
        raise ValueError("Not enough articles/features for the requested LSA dimensions.")
    svd = TruncatedSVD(n_components=dimensions, random_state=seed)
    normalizer = Normalizer(copy=False)
    dense = normalizer.fit_transform(svd.fit_transform(sparse))
    return dict(tfidf=tfidf, svd=svd, normalizer=normalizer,
                sparse=sparse, dense=dense,
                valid=np.asarray(sparse.getnnz(axis=1) > 0))


def evaluate_labels(dense, reference_sample, labels, sample_indices):
    """Use fixed samples for silhouettes and all LSA vectors for CH and DB."""
    sample_labels = np.asarray(labels)[sample_indices]
    if not 1 < len(np.unique(sample_labels)) < len(sample_labels):
        raise ValueError("The evaluation sample must contain at least two clusters.")
    counts = np.bincount(labels)
    return {
        "Silhouette TF-IDF cosine": silhouette_score(
            reference_sample, sample_labels, metric="cosine"),
        "Silhouette LSA Euclidean": silhouette_score(
            dense[sample_indices], sample_labels, metric="euclidean"),
        "Calinski-Harabasz LSA": calinski_harabasz_score(dense, labels),
        "Davies-Bouldin LSA": davies_bouldin_score(dense, labels),
        "Min cluster share": counts.min() / counts.sum(),
        "Max cluster share": counts.max() / counts.sum(),
    }


def search_clusters(dense, reference_sample, sample_indices,
                    k_values=tuple(range(2, 17)), seeds=(42, 17, 93), n_init=10):
    """Compare K across seeds; keep the first seed's exact fitted estimator.

    Mean cosine silhouette is measured BEFORE LSA, in the same topic TF-IDF
    space for every K. LSA silhouette is diagnostic rather than the objective.
    CH and DB use all normalized LSA vectors, with mean/std across seeds.
    Stability is the mean pairwise ARI of cluster assignments across seeds.
    """
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least two distinct seeds to measure stability.")
    if any(k < 2 or k >= len(dense) for k in k_values):
        raise ValueError("Every K must be between 2 and n_articles - 1.")
    runs, summary, models = [], [], {}
    for k in k_values:
        labels = []
        current = []
        for seed in seeds:
            model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(dense)
            labels.append(model.labels_)
            scores = evaluate_labels(dense, reference_sample, model.labels_, sample_indices)
            row = dict(K=k, Seed=seed, Inertia=model.inertia_, **scores)
            runs.append(row)
            current.append(row)
            if seed == seeds[0]:
                models[k] = model
        t = pd.DataFrame(current)
        summary.append({
            "K": k,
            "Silhouette TF-IDF cosine": t["Silhouette TF-IDF cosine"].mean(),
            "Silhouette TF-IDF std": t["Silhouette TF-IDF cosine"].std(ddof=0),
            "Silhouette LSA Euclidean": t["Silhouette LSA Euclidean"].mean(),
            "Calinski-Harabasz LSA": t["Calinski-Harabasz LSA"].mean(),
            "Calinski-Harabasz std": t["Calinski-Harabasz LSA"].std(ddof=0),
            "Davies-Bouldin LSA": t["Davies-Bouldin LSA"].mean(),
            "Davies-Bouldin std": t["Davies-Bouldin LSA"].std(ddof=0),
            "Stability ARI": np.mean([adjusted_rand_score(a, b)
                                      for a, b in combinations(labels, 2)]),
            "Min cluster share": t["Min cluster share"].min(),
            "Max cluster share": t["Max cluster share"].max(),
            "Inertia": t["Inertia"].mean(),
        })
        print(f"K={k:2d}: cosine={summary[-1]['Silhouette TF-IDF cosine']:.4f}, "
              f"CH={summary[-1]['Calinski-Harabasz LSA']:.2f}, "
              f"DB={summary[-1]['Davies-Bouldin LSA']:.4f}, "
              f"stability={summary[-1]['Stability ARI']:.3f}", flush=True)
    return pd.DataFrame(summary).set_index("K"), pd.DataFrame(runs), models


def keyword_table(sparse, terms, labels, top_n=15, min_document_share=0.01):
    """Rank salient words/phrases using mean TF-IDF times log(1 + lift).

    Document support prevents a rare phrase from defining an entire cluster.
    We report the underlying weights and support, not just keyword strings.
    """
    labels = np.asarray(labels)
    global_mean = np.asarray(sparse.mean(axis=0)).ravel()
    rows = []
    for cluster in np.unique(labels):
        subset = sparse[labels == cluster]
        means = np.asarray(subset.mean(axis=0)).ravel()
        support = np.asarray(subset.getnnz(axis=0)).ravel() / subset.shape[0]
        score = means * np.log1p(means / np.maximum(global_mean, 1e-12))
        score[support < min_document_share] = 0
        candidates = np.flatnonzero(score > 0)
        order = candidates[np.argsort(-score[candidates], kind="stable")[:top_n]]
        for rank, index in enumerate(order, 1):
            rows.append({"cluster": int(cluster), "rank": rank,
                         "term": terms[index], "mean_tfidf": float(means[index]),
                         "contrast_score": float(score[index]),
                         "document_share": float(support[index])})
    return pd.DataFrame(rows)


def article_tables(frame, dense, model, representatives=3, boundary_examples=2):
    """Find central and ambiguous articles in the fitted LSA geometry.

    Margin = distance to second-closest center minus distance to own center.
    It is a geometric diagnostic, never a calibrated probability/confidence.
    """
    distances = model.transform(dense)
    labels = model.labels_
    closest_two = np.sort(distances, axis=1)[:, :2]
    articles = frame[["raw_id", "title"]].reset_index(drop=True).copy()
    articles["cluster"] = labels
    articles["distance_to_center"] = distances[np.arange(len(labels)), labels]
    articles["boundary_margin"] = closest_two[:, 1] - closest_two[:, 0]
    articles["distance_percentile"] = articles.groupby("cluster")[
        "distance_to_center"].rank(method="average", pct=True)
    central = (articles.sort_values(["cluster", "distance_to_center", "raw_id"])
               .groupby("cluster", sort=True).head(representatives).copy())
    boundary = (articles.sort_values(["cluster", "boundary_margin", "raw_id"])
                .groupby("cluster", sort=True).head(boundary_examples).copy())
    central["rank"] = central.groupby("cluster").cumcount() + 1
    boundary["rank"] = boundary.groupby("cluster").cumcount() + 1
    return articles, central, boundary


def fitted_pipeline(features, model):
    """Bundle all fitted stages; input remains notebook-3 clean_text."""
    return Pipeline([("tfidf", features["tfidf"]), ("lsa", features["svd"]),
                     ("normalizer", features["normalizer"]), ("kmeans", model)])


def predict_topics(pipeline, clean_texts):
    """Return -1 for texts with no known features, rather than an arbitrary topic."""
    sparse = pipeline.named_steps["tfidf"].transform(clean_texts)
    valid = np.asarray(sparse.getnnz(axis=1) > 0)
    predictions = np.full(sparse.shape[0], -1, dtype=int)
    if valid.any():
        dense = pipeline.named_steps["normalizer"].transform(
            pipeline.named_steps["lsa"].transform(sparse[valid]))
        predictions[valid] = pipeline.named_steps["kmeans"].predict(dense)
    return predictions
