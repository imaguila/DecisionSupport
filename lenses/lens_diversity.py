"""
Diversity Lens Module.

Structures candidate solution sets into clusters using distance-based or
density-based unsupervised learning algorithms (K-Medoids, K-Means,
Agglomerative Hierarchical Clustering, or HDBSCAN).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Optional dependencies for enhanced clustering algorithms
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
# UI RENDERING
# =====================================================


def render_params(
    dataset: Dict[str, Any], working_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Renders Streamlit UI controls for diversity clustering algorithms.

    Parameters
    ----------
    dataset : Dict[str, Any]
        Global dataset configuration containing metric and indicator keys.
    working_df : pd.DataFrame
        Current working solution space DataFrame.

    Returns
    -------
    Dict[str, Any]
        Dictionary of selected clustering parameters.
    """
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    params: Dict[str, Any] = {}
    max_n = max(len(working_df), 1)

    if len(dimensions) < 2:
        st.info("At least two dimensions are required for clustering.")
        params["method"] = "K-Medoids"
        params["cluster_metrics"] = []
        return params

    params["method"] = st.selectbox(
        "Clustering Method",
        ["K-Medoids", "K-Means", "Agglomerative", "HDBSCAN"],
        key="div_method",
    )

    default_cluster_metrics = dimensions[: min(2, len(dimensions))]

    params["cluster_metrics"] = st.multiselect(
        "Metrics for Clustering",
        dimensions,
        default=default_cluster_metrics,
        key="div_cluster_metrics",
    )

    if params["method"] in ["K-Medoids", "K-Means"]:
        params["k_mode"] = st.radio(
            "Number of Groups",
            ["Auto", "Manual"],
            horizontal=True,
            key="div_k_mode",
        )

        if params["k_mode"] == "Manual":
            max_k = max(2, min(10, max_n))
            default_k = min(3, max_k)
            params["k"] = st.slider(
                "k Groups", 2, max_k, default_k, key="div_k"
            )
        else:
            st.caption(
                "Auto mode selects k using silhouette score maximization."
            )

    elif params["method"] == "Agglomerative":
        params["agglomerative_mode"] = st.radio(
            "Hierarchy Cut Mode",
            ["Number of Groups", "Distance Cut"],
            horizontal=True,
            key="div_agglomerative_mode",
        )

        if params["agglomerative_mode"] == "Number of Groups":
            params["k_mode"] = st.radio(
                "Number of Groups",
                ["Auto", "Manual"],
                horizontal=True,
                key="div_agg_k_mode",
            )

            if params["k_mode"] == "Manual":
                max_k = max(2, min(10, max_n))
                default_k = min(3, max_k)
                params["k"] = st.slider(
                    "k Groups", 2, max_k, default_k, key="div_agg_k"
                )
            else:
                st.caption(
                    "Auto mode selects the dendrogram cut with the best silhouette score."
                )
        else:
            params["distance_threshold"] = st.slider(
                "Distance Threshold",
                0.10,
                10.00,
                2.00,
                0.10,
                key="div_agg_distance_threshold",
            )
            st.caption(
                "Distance Cut builds the hierarchy and cuts it at the selected distance threshold."
            )

    elif params["method"] == "HDBSCAN":
        params["cluster_size_mode"] = st.radio(
            "Cluster Size",
            ["Auto", "Manual"],
            horizontal=True,
            key="div_hdbscan_size_mode",
        )

        if params["cluster_size_mode"] == "Auto":
            params["granularity"] = st.selectbox(
                "Cluster Granularity",
                ["Small (~5%)", "Medium (~10%)", "Large (~20%)"],
                index=1,
                key="div_hdbscan_granularity",
            )
        else:
            default_min_size = max(2, int(0.10 * max_n))
            params["min_cluster_size"] = st.slider(
                "Minimum Cluster Size",
                2,
                max(2, max_n),
                default_min_size,
                key="div_hdbscan_min_cluster_size",
            )

        params["exclude_noise"] = st.checkbox(
            "Exclude noise solutions",
            value=True,
            key="div_hdbscan_exclude_noise",
        )
        st.caption(
            "If HDBSCAN returns mostly noise, reduce cluster size or disable noise exclusion."
        )

    st.caption(
        "Diversity structures candidate solutions into spatial clusters instead of preference ranking."
    )

    return params


# =====================================================
# HELPER FUNCTIONS
# =====================================================


def _valid_numeric_metrics(df: pd.DataFrame, metrics: List[str]) -> List[str]:
    """
    Filters metrics present in DataFrame that are strictly numeric.
    """
    return [
        m
        for m in metrics
        if m in df.columns and pd.api.types.is_numeric_dtype(df[m])
    ]


def _prepare_matrix(df: pd.DataFrame, metrics: List[str]) -> np.ndarray:
    """
    Imputes missing values and standardizes features using Z-score scaling.
    """
    x = df[metrics].copy()
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def _build_partition_model(
    method: str, k: int
) -> Union[KMeans, AgglomerativeClustering, Any]:
    """
    Instantiates specified partition clustering model instance.
    """
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
    """
    Determines optimal number of clusters k via silhouette score maximization.
    """
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
    """
    Fits partition clustering model and returns assigned cluster labels.
    """
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
    """
    Fits HDBSCAN density model if available.
    """
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
    """
    Fits Agglomerative clustering cut at fixed distance threshold.
    """
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
    """
    Safely calculates silhouette score if valid number of clusters exist.
    """
    unique_labels = set(labels)
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
    """
    Attaches cluster IDs, labels, sizes, and metadata to output DataFrame.
    """
    res = result.copy()
    res["cluster"] = labels
    res["cluster_str"] = res["cluster"].astype(str).replace("-1", "Noise")

    cluster_sizes = res.groupby("cluster_str")["id"].transform("size")
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
    df: pd.DataFrame, params: Dict[str, Any], dataset: Dict[str, Any]
) -> pd.DataFrame:
    """
    Applies selected diversity lens method to structure DataFrame into clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Input working solution space DataFrame.
    params : Dict[str, Any]
        Clustering configuration parameters.
    dataset : Dict[str, Any]
        Global context dataset metadata.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with cluster assignments and metadata.
    """
    if df is None or df.empty or len(df) < 2:
        return df

    result = df.copy()
    dimensions = dataset.get("metrics", []) + dataset.get(
        "selected_indicators", []
    )
    method = params.get("method", "K-Medoids")
    cluster_metrics = params.get("cluster_metrics", dimensions)
    cluster_metrics = _valid_numeric_metrics(result, cluster_metrics)

    if len(cluster_metrics) < 2:
        return result

    x_scaled = _prepare_matrix(result, cluster_metrics)

    # ----------------------------------------------------
    # K-Medoids / K-Means
    # ----------------------------------------------------
    if method in ["K-Medoids", "K-Means"]:
        k_mode = params.get("k_mode", "Auto")

        if k_mode == "Manual":
            k = max(2, min(params.get("k", 2), len(result)))
            silhouette = None
        else:
            k, silhouette = _compute_auto_k(x_scaled, method)
            if k < 2:
                return result

        labels, method_used = _fit_partition_clustering(x_scaled, method, k)
        result = _add_cluster_labels(
            result, labels, method_used, cluster_metrics
        )
        result["diversity_k"] = k
        if silhouette is not None:
            result["diversity_silhouette"] = silhouette

        return result

    # ----------------------------------------------------
    # Agglomerative
    # ----------------------------------------------------
    if method == "Agglomerative":
        agglomerative_mode = params.get(
            "agglomerative_mode", "Number of Groups"
        )

        if agglomerative_mode == "Distance Cut":
            dist_thresh = params.get("distance_threshold", 2.0)
            labels, method_used = _fit_agglomerative_distance_cut(
                x_scaled, dist_thresh
            )
            result = _add_cluster_labels(
                result, labels, method_used, cluster_metrics
            )
            result["diversity_distance_threshold"] = dist_thresh

            silhouette = _compute_silhouette_if_valid(x_scaled, labels)
            if silhouette is not None:
                result["diversity_silhouette"] = silhouette

            return result

        k_mode = params.get("k_mode", "Auto")
        if k_mode == "Manual":
            k = max(2, min(params.get("k", 2), len(result)))
            silhouette = None
        else:
            k, silhouette = _compute_auto_k(x_scaled, method)
            if k < 2:
                return result

        labels, method_used = _fit_partition_clustering(x_scaled, method, k)
        result = _add_cluster_labels(
            result, labels, method_used, cluster_metrics
        )
        result["diversity_k"] = k
        if silhouette is not None:
            result["diversity_silhouette"] = silhouette

        return result

    # ----------------------------------------------------
    # HDBSCAN
    # ----------------------------------------------------
    if method == "HDBSCAN":
        n = len(result)
        size_mode = params.get("cluster_size_mode", "Auto")

        if size_mode == "Manual":
            min_cluster_size = params.get(
                "min_cluster_size", max(2, int(0.1 * n))
            )
        else:
            granularity = params.get("granularity", "Medium (~10%)")
            if granularity == "Small (~5%)":
                min_cluster_size = max(2, int(0.05 * n))
            elif granularity == "Large (~20%)":
                min_cluster_size = max(2, int(0.20 * n))
            else:
                min_cluster_size = max(2, int(0.10 * n))

        labels, method_used = _fit_hdbscan(x_scaled, min_cluster_size)
        result = _add_cluster_labels(
            result, labels, method_used, cluster_metrics
        )
        result["diversity_min_cluster_size"] = min_cluster_size

        if params.get("exclude_noise", True):
            filtered = result[result["cluster"] != -1].copy()
            if filtered.empty:
                result["diversity_warning"] = (
                    "All solutions were classified as noise. "
                    "Noise exclusion was not applied."
                )
                return result
            return filtered

        return result

    return result


# =====================================================
# FEEDBACK UI
# =====================================================


def _safe_first_value(df: pd.DataFrame, column: str) -> Any:
    """
    Extracts first non-null value from given DataFrame column if present.
    """
    if column not in df.columns:
        return None
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else None


def render_feedback(lens_df: pd.DataFrame) -> None:
    """
    Displays UI summary metrics and feedback for applied clustering lens.
    """
    if lens_df is None:
        st.warning("No clustering result is available.")
        return

    if lens_df.empty:
        st.warning(
            "The clustering lens returned an empty subset. "
            "Try reducing HDBSCAN minimum cluster size or disabling noise exclusion."
        )
        return

    warning_value = _safe_first_value(lens_df, "diversity_warning")
    if warning_value is not None:
        st.warning(warning_value)

    n_clusters = _safe_first_value(lens_df, "diversity_n_clusters")
    if n_clusters is not None:
        st.info(f"Clusters detected: **{int(n_clusters)}**")

    k_value = _safe_first_value(lens_df, "diversity_k")
    if k_value is not None:
        st.caption(f"Selected k: **{int(k_value)}**")

    silhouette_val = _safe_first_value(lens_df, "diversity_silhouette")
    if silhouette_val is not None:
        st.caption(f"Silhouette score: **{silhouette_val:.3f}**")

    min_cluster_size = _safe_first_value(lens_df, "diversity_min_cluster_size")
    if min_cluster_size is not None:
        st.caption(f"Minimum cluster size: **{int(min_cluster_size)}**")

    dist_thresh = _safe_first_value(lens_df, "diversity_distance_threshold")
    if dist_thresh is not None:
        st.caption(f"Distance threshold: **{float(dist_thresh):.2f}**")

    noise_count = _safe_first_value(lens_df, "diversity_noise_count")
    if noise_count is not None and int(noise_count) > 0:
        st.caption(f"Noise solutions detected: **{int(noise_count)}**")