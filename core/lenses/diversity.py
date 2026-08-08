"""
Diversity Lens Module (Headless Core).

Structures candidate solution sets into clusters using distance-based or
density-based unsupervised learning algorithms (K-Medoids, K-Means,
Agglomerative Hierarchical Clustering, or HDBSCAN) and filters solutions by 
the user-selected target cluster (SOI).

Pure Python & Pandas module — Framework agnostic.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Optional clustering dependencies
try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    HDBSCAN = None

logger = logging.getLogger(__name__)


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _valid_numeric_metrics(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    """Filters metrics present in DataFrame that are strictly numeric."""
    return [
        m
        for m in metrics
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m])
    ]


def _prepare_matrix(df: pd.DataFrame, metrics: List[str]) -> np.ndarray:
    """Imputes missing values and standardizes features using Z-score scaling."""
    x = df[metrics].copy()
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def _build_partition_model(
    method: str, k: int
) -> Union[KMeans, AgglomerativeClustering, Any]:
    """Instantiates specified partition clustering model instance."""
    if method == "K-Medoids":
        if KMedoids is not None:
            return KMedoids(n_clusters=k, method="pam", random_state=123)
        logger.warning(
            "scikit-learn-extra KMedoids not installed. Falling back to KMeans."
        )
        return KMeans(n_clusters=k, random_state=123, n_init=10)

    if method == "K-Means":
        return KMeans(n_clusters=k, random_state=123, n_init=10)

    if method == "Agglomerative":
        return AgglomerativeClustering(n_clusters=k)

    return KMeans(n_clusters=k, random_state=123, n_init=10)


def _compute_auto_k(
    x_scaled: np.ndarray, method: str, max_k: int = 10
) -> Tuple[int, Optional[float]]:
    """Determines optimal number of clusters k via silhouette score maximization."""
    n = len(x_scaled)
    if n < 3:
        return 1, None

    best_k = 2
    best_score = -1.0
    upper_k = min(max_k, n - 1)

    for k in range(2, upper_k + 1):
        try:
            model = _build_partition_model(method, k)
            labels = model.fit_predict(x_scaled)
            unique_labels = set(labels)

            if 1 < len(unique_labels) < n:
                score = silhouette_score(x_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        except Exception as err:
            logger.debug("Silhouette evaluation failed for k=%d: %s", k, err)

    return best_k, (best_score if best_score != -1.0 else None)


def _fit_partition_clustering(
    x_scaled: np.ndarray, method: str, k: int
) -> Tuple[np.ndarray, str]:
    """Fits partition clustering model and returns assigned cluster labels."""
    model = _build_partition_model(method, k)
    labels = model.fit_predict(x_scaled)

    method_used = (
        "K-Means fallback"
        if (method == "K-Medoids" and KMedoids is None)
        else method
    )
    return labels, method_used


def _fit_hdbscan(
    x_scaled: np.ndarray, min_cluster_size: int
) -> Tuple[np.ndarray, str]:
    """Fits HDBSCAN density model if available."""
    if HDBSCAN is None:
        logger.warning("HDBSCAN module not installed.")
        labels = np.zeros(len(x_scaled), dtype=int)
        return labels, "HDBSCAN unavailable"

    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(x_scaled)
    return labels, "HDBSCAN"


def _fit_agglomerative_distance_cut(
    x_scaled: np.ndarray, distance_threshold: float
) -> Tuple[np.ndarray, str]:
    """Fits Agglomerative clustering cut at fixed distance threshold."""
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        compute_full_tree=True,
    )
    labels = model.fit_predict(x_scaled)
    return labels, "Agglomerative distance cut"


def _compute_silhouette_if_valid(
    x_scaled: np.ndarray, labels: np.ndarray
) -> Optional[float]:
    """Safely calculates silhouette score if valid number of clusters exist."""
    unique_labels = set(labels) - {-1}
    n = len(labels)

    if len(unique_labels) <= 1 or len(unique_labels) >= n:
        return None

    try:
        return float(silhouette_score(x_scaled, labels))
    except Exception:
        return None


def _add_cluster_labels(
    result: pd.DataFrame,
    labels: np.ndarray,
    method_used: str,
    metrics_used: List[str],
) -> pd.DataFrame:
    """Attaches cluster IDs, labels, sizes, and metadata to output DataFrame."""
    res = result.copy()
    res["cluster"] = labels
    res["cluster_str"] = res["cluster"].astype(str).replace("-1", "Noise")

    id_col = "id" if "id" in res.columns else res.index
    cluster_sizes = res.groupby("cluster_str")[id_col].transform("size")
    res["group_label"] = (
        "Cluster " + res["cluster_str"] + " (n=" + cluster_sizes.astype(str) + ")"
    )

    n_clusters = (
        res["cluster"]
        .dropna()
        .astype(int)
        .loc[lambda v: v != -1]
        .nunique()
    )
    noise_count = int(res["cluster"].eq(-1).sum())

    res["diversity_method"] = method_used
    res["diversity_metrics"] = ", ".join(metrics_used)
    res["diversity_n_clusters"] = n_clusters
    res["diversity_noise_count"] = noise_count

    return res


# =====================================================
# MAIN PIPELINE ENTRY POINT
# =====================================================


def apply(
    df: pd.DataFrame,
    params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Applies selected diversity lens method to structure and filter DataFrame into clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Clustering setup parameters including `selected_cluster` (SOI).
    context : Dict[str, Any], optional
        Global dataset context metadata.

    Returns
    -------
    pd.DataFrame
        Enriched and optionally filtered DataFrame for the SOI cluster.
    """
    if df is None or df.empty or len(df) < 2:
        return df

    result = df.copy()
    metrics_from_context = (context or {}).get("metrics", []) + (context or {}).get("selected_indicators", [])
    method = params.get("method", "K-Medoids")
    cluster_metrics = params.get("cluster_metrics", metrics_from_context)
    cluster_metrics = _valid_numeric_metrics(result, cluster_metrics)

    if len(cluster_metrics) < 2:
        return result

    x_scaled = _prepare_matrix(result, cluster_metrics)
    silhouette: Optional[float] = None

    # 1. Clustering execution
    if method in ["K-Medoids", "K-Means"]:
        k_mode = params.get("k_mode", "Auto")

        if k_mode == "Manual":
            k = max(2, min(params.get("k", 2), len(result)))
            silhouette = _compute_silhouette_if_valid(x_scaled, _build_partition_model(method, k).fit_predict(x_scaled))
        else:
            k, silhouette = _compute_auto_k(x_scaled, method)
            if k < 2:
                return result

        labels, method_used = _fit_partition_clustering(x_scaled, method, k)
        result = _add_cluster_labels(result, labels, method_used, cluster_metrics)
        result["diversity_k"] = k

    elif method == "Agglomerative":
        agglomerative_mode = params.get("agglomerative_mode", "Number of Groups")

        if agglomerative_mode == "Distance Cut":
            dist_thresh = params.get("distance_threshold", 2.0)
            labels, method_used = _fit_agglomerative_distance_cut(x_scaled, dist_thresh)
            result = _add_cluster_labels(result, labels, method_used, cluster_metrics)
            result["diversity_distance_threshold"] = dist_thresh
            silhouette = _compute_silhouette_if_valid(x_scaled, labels)
        else:
            k_mode = params.get("k_mode", "Auto")
            if k_mode == "Manual":
                k = max(2, min(params.get("k", 2), len(result)))
                silhouette = _compute_silhouette_if_valid(x_scaled, _build_partition_model(method, k).fit_predict(x_scaled))
            else:
                k, silhouette = _compute_auto_k(x_scaled, method)
                if k < 2:
                    return result

            labels, method_used = _fit_partition_clustering(x_scaled, method, k)
            result = _add_cluster_labels(result, labels, method_used, cluster_metrics)
            result["diversity_k"] = k

    elif method == "HDBSCAN":
        n = len(result)
        size_mode = params.get("cluster_size_mode", "Auto")

        if size_mode == "Manual":
            min_cluster_size = params.get("min_cluster_size", max(2, int(0.1 * n)))
        else:
            granularity = params.get("granularity", "Medium (~10%)")
            if granularity == "Small (~5%)":
                min_cluster_size = max(2, int(0.05 * n))
            elif granularity == "Large (~20%)":
                min_cluster_size = max(2, int(0.20 * n))
            else:
                min_cluster_size = max(2, int(0.10 * n))

        labels, method_used = _fit_hdbscan(x_scaled, min_cluster_size)
        result = _add_cluster_labels(result, labels, method_used, cluster_metrics)
        result["diversity_min_cluster_size"] = min_cluster_size
        silhouette = _compute_silhouette_if_valid(x_scaled, labels)

        if params.get("exclude_noise", True) and not params.get("selected_cluster"):
            filtered = result[result["cluster"] != -1].copy()
            if not filtered.empty:
                result = filtered

    # Attach quality metrics to df.attrs for UI feedback
    if silhouette is not None:
        result["diversity_silhouette"] = silhouette
        result.attrs["silhouette_score"] = silhouette
    
    n_clusters_detected = result["cluster_str"].replace("Noise", np.nan).dropna().nunique()
    result.attrs["n_clusters"] = n_clusters_detected

    # 2. SOI Filtering by Selected Cluster
    selected_cluster = params.get("selected_cluster", "All")
    if selected_cluster and selected_cluster != "All" and "cluster_str" in result.columns:
        result = result[result["cluster_str"] == str(selected_cluster)].copy()

    return result