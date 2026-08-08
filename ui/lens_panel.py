"""
Lens Panel UI Module.

Handles Streamlit UI controls for all analytical lenses, delegates computation,
embeds SOI saving within the lens expander, and renders analytics feedback.
"""

from typing import Any, Dict, Tuple, List
import pandas as pd
import streamlit as st

import core.lenses.diversity as diversity_core
from core.lenses.registry import get_lens_names
from ui.soi_panel import render_soi_saver


# =====================================================
# CONTROLES INDIVIDUALES DE LENTES
# =====================================================

def render_manual_controls(working_df: pd.DataFrame) -> Dict[str, Any]:
    if working_df is None or working_df.empty or "id" not in working_df.columns:
        st.warning("No solution IDs available for manual selection.")
        return {"selected_ids": []}

    valid_ids = working_df["id"].dropna().astype(int).tolist()
    selected = st.multiselect(
        "Pick solutions by ID",
        options=valid_ids,
        key="manual_lens_ids",
        help="Select explicit solutions to isolate.",
    )
    return {"method": "Manual Selection", "selected_ids": selected}


def render_preference_controls(dimensions: list, max_solutions: int) -> Dict[str, Any]:
    method = st.selectbox(
        "Scoring Method",
        ["Weighted Sum", "TOPSIS", "VIKOR", "Reference Point"],
        key="pref_method",
    )
    st.caption("Criteria currently carry equal relative weighting.")

    maximize = st.multiselect("Metrics to Maximize", dimensions, key="pref_maximize")
    minimize_options = [d for d in dimensions if d not in maximize]
    minimize = st.multiselect("Metrics to Minimize", minimize_options, key="pref_minimize")

    top_n = st.slider("Top N Solutions", 1, max(max_solutions, 1), min(5, max_solutions or 1), key="pref_top_n")

    return {
        "method": method,
        "maximize": maximize,
        "minimize": minimize,
        "top_n": top_n,
    }


def render_efficiency_controls(dimensions: list, working_df: pd.DataFrame) -> Dict[str, Any]:
    max_n = max(len(working_df) if working_df is not None else 0, 1)
    default_n = min(5, max_n)

    if len(dimensions) < 2:
        st.info("At least two dimensions are required for the Efficiency lens.")
        return {
            "method": "Benefit/Cost Ratio",
            "benefit": None,
            "cost": None,
            "top_n": default_n,
        }

    method = st.selectbox(
        "Efficiency Method",
        [
            "Benefit/Cost Ratio",
            "Normalized Ratio",
            "Distance to Ideal",
            "Composite Cost Ratio",
        ],
        key="eff_method",
    )

    benefit = st.selectbox("Benefit Metric", dimensions, key="eff_benefit")
    cost_options = [d for d in dimensions if d != benefit]

    if method == "Composite Cost Ratio":
        cost = st.multiselect(
            "Cost Metrics",
            cost_options,
            default=cost_options[: min(2, len(cost_options))],
            key="eff_costs",
        )
    else:
        cost = st.selectbox("Cost Metric", cost_options, key="eff_cost")

    top_n = st.slider("Top N Solutions", 1, max_n, default_n, key="eff_top_n")

    return {
        "method": method,
        "benefit": benefit,
        "cost": cost,
        "top_n": top_n,
    }


def render_indicator_controls(dimensions: list, working_df: pd.DataFrame) -> Dict[str, Any]:
    max_n = max(len(working_df) if working_df is not None else 0, 1)
    default_n = min(5, max_n)

    if not dimensions:
        st.info("No dimensions available for Indicator Lens.")
        return {
            "method": "Top-N Matches",
            "maximize": [],
            "minimize": [],
            "top_n": default_n,
            "match_filter": "All",
            "target_matches": 1,
        }

    method = st.selectbox(
        "Indicator Method",
        ["Top-N Matches", "Non-dominated"],
        key="indicator_method",
    )

    maximize = st.multiselect("Dimensions to Maximize", dimensions, key="indicator_maximize")
    minimize_options = [c for c in dimensions if c not in maximize]
    minimize = st.multiselect("Dimensions to Minimize", minimize_options, key="indicator_minimize")

    params: Dict[str, Any] = {
        "method": method,
        "maximize": maximize,
        "minimize": minimize,
    }

    if method == "Top-N Matches":
        params["top_n"] = st.slider("Top N per Dimension", 1, max_n, default_n, key="indicator_top_n")

        total_selected = len(maximize) + len(minimize)
        if total_selected > 1:
            st.markdown("##### 🎯 Target Match Group")
            match_filter_label = st.selectbox(
                "Filter Match Group",
                [
                    "All matches (>0)",
                    "Highest matches only",
                    "At least N matches",
                    "Exact N matches",
                ],
                key="indicator_match_filter",
            )

            filter_map = {
                "All matches (>0)": "All",
                "Highest matches only": "Highest",
                "At least N matches": "At least",
                "Exact N matches": "Exact",
            }
            params["match_filter"] = filter_map[match_filter_label]

            if params["match_filter"] in ["At least", "Exact"]:
                params["target_matches"] = st.slider(
                    "Matches Target (N)",
                    min_value=1,
                    max_value=total_selected,
                    value=min(2, total_selected),
                    key="indicator_target_matches",
                )
            else:
                params["target_matches"] = 1
        else:
            params["match_filter"] = "All"
            params["target_matches"] = 1
    else:
        params["top_n"] = None
        params["match_filter"] = "All"
        params["target_matches"] = 1

    return params


def render_diversity_controls(dimensions: list, working_df: pd.DataFrame) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    max_n = max(len(working_df) if working_df is not None else 0, 1)

    if len(dimensions) < 2:
        st.info("At least two dimensions are required for clustering.")
        return {"method": "K-Medoids", "cluster_metrics": [], "selected_cluster": "All"}

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

    def _render_inline_cluster_metrics(curr_params: dict):
        if len(curr_params.get("cluster_metrics", [])) < 2 or working_df is None or len(working_df) < 2:
            return

        p = curr_params.copy()
        p["selected_cluster"] = "All"
        preview_df = diversity_core.apply(working_df, p)

        if preview_df is None or preview_df.empty:
            return

        sil_score = preview_df.attrs.get("silhouette_score", None)
        if sil_score is None and "diversity_silhouette" in preview_df.columns:
            sil_vals = preview_df["diversity_silhouette"].dropna()
            if not sil_vals.empty:
                sil_score = float(sil_vals.iloc[0])

        n_clusters = preview_df.attrs.get("n_clusters", None)
        if n_clusters is None and "cluster_str" in preview_df.columns:
            clusters = [c for c in preview_df["cluster_str"].unique() if c != "Noise"]
            n_clusters = len(clusters)

        if sil_score is not None:
            if sil_score > 0.5:
                quality = "🟢 High"
            elif sil_score > 0.25:
                quality = "🟡 Moderate"
            else:
                quality = "🔴 Weak"
            sil_display = f"**{sil_score:.2f}** ({quality})"
        else:
            sil_display = "**N/A**"

        st.markdown(f"**Groups (k):** **`{n_clusters if n_clusters else 'N/A'}`** ")
        st.markdown(f" **Silhouette:** {sil_display}")

    if params["method"] in ["K-Medoids", "K-Means"]:
        params["k_mode"] = st.radio("Number of Groups", ["Auto", "Manual"], horizontal=True, key="div_k_mode")
        if params["k_mode"] == "Manual":
            max_k = max(2, min(10, max_n))
            params["k"] = st.slider("k Groups", 2, max_k, min(3, max_k), key="div_k")
        _render_inline_cluster_metrics(params)

    elif params["method"] == "Agglomerative":
        params["agglomerative_mode"] = st.radio(
            "Hierarchy Cut Mode",
            ["Number of Groups", "Distance Cut"],
            horizontal=True,
            key="div_agglomerative_mode",
        )
        if params["agglomerative_mode"] == "Number of Groups":
            params["k_mode"] = st.radio("Number of Groups", ["Auto", "Manual"], horizontal=True, key="div_agg_k_mode")
            if params["k_mode"] == "Manual":
                max_k = max(2, min(10, max_n))
                params["k"] = st.slider("k Groups", 2, max_k, min(3, max_k), key="div_agg_k")
        else:
            params["distance_threshold"] = st.slider(
                "Distance Threshold", 0.10, 10.00, 2.00, 0.10, key="div_agg_distance_threshold"
            )
        _render_inline_cluster_metrics(params)

    elif params["method"] == "HDBSCAN":
        params["cluster_size_mode"] = st.radio(
            "Cluster Size", ["Auto", "Manual"], horizontal=True, key="div_hdbscan_size_mode"
        )
        if params["cluster_size_mode"] == "Auto":
            params["granularity"] = st.selectbox(
                "Cluster Granularity",
                ["Small (~5%)", "Medium (~10%)", "Large (~20%)"],
                index=1,
                key="div_hdbscan_granularity",
            )
        else:
            params["min_cluster_size"] = st.slider(
                "Minimum Cluster Size", 2, max(2, max_n), max(2, int(0.10 * max_n)), key="div_hdbscan_min_cluster_size"
            )
        params["exclude_noise"] = st.checkbox("Exclude noise solutions", value=True, key="div_hdbscan_exclude_noise")
        _render_inline_cluster_metrics(params)

    if len(params["cluster_metrics"]) >= 2 and len(working_df) >= 2:
        st.markdown("##### 🎯 Target Cluster (SOI)")
        preview_params = params.copy()
        preview_params["selected_cluster"] = "All"
        preview_df = diversity_core.apply(working_df, preview_params)

        if "cluster_str" in preview_df.columns:
            counts = preview_df["cluster_str"].value_counts().to_dict()
            options = ["All"] + sorted(list(counts.keys()), key=lambda x: (x == "Noise", int(x) if x.isdigit() else 999))
            
            format_func = lambda opt: "All Clusters" if opt == "All" else f"Cluster {opt} (n={counts[opt]})"
            params["selected_cluster"] = st.selectbox(
                "Select SOI Cluster",
                options,
                format_func=format_func,
                key="div_selected_cluster",
            )
        else:
            params["selected_cluster"] = "All"
    else:
        params["selected_cluster"] = "All"

    return params


# =====================================================
# EXPANDER DE LENTES & SOI SAVER INTEGRADO
# =====================================================

def render_lens_panel(engine: Any) -> Tuple[str, Dict[str, Any]]:
    """Renders lens controls and embeds SOI saving within the same expander."""
    with st.sidebar.expander("🧭 Solution of Interest (Lens)", expanded=False):
        lens_names = get_lens_names()
        
        active_lens_name = st.selectbox(
            "Select Analytical Lens",
            options=lens_names,
            index=0,
            key="active_lens_name",
        )

        params: Dict[str, Any] = {}
        dimensions = engine.framing_dimensions
        working_df = engine.framed_df
        working_len = len(working_df) if working_df is not None else 0

        if active_lens_name == "Manual Selection":
            params = render_manual_controls(working_df)
        elif active_lens_name == "Preference":
            params = render_preference_controls(dimensions, working_len)
        elif active_lens_name == "Efficiency":
            params = render_efficiency_controls(dimensions, working_df)
        elif active_lens_name == "Indicators":
            params = render_indicator_controls(dimensions, working_df)
        elif active_lens_name == "Diversity":
            params = render_diversity_controls(dimensions, working_df)
        elif active_lens_name == "Consensus":
            params = render_consensus_controls(engine)

        # 1. Aplicar inmediatamente la Lente en el motor
        engine.apply_lens(active_lens_name, params)

        # 2. Renderizar Guardado de SOI DENTRO del mismo expander si hay Lente activa
        if active_lens_name and active_lens_name != "None":
            render_soi_saver(engine, active_lens_name, params)

        return active_lens_name, params


def render_lens_feedback(active_lens_name: str, active_df: pd.DataFrame) -> None:
    """Renders contextual feedback dynamically for ANY active Lens."""
    if not active_lens_name or active_lens_name == "None" or active_df is None:
        return

    count = len(active_df)

    candidate_cols = [
        f"{active_lens_name.lower().split()[0]}_method",
        "diversity_method",
        "preference_method",
        "efficiency_method",
        "indicator_method",
    ]

    method = None
    for col in candidate_cols:
        if col in active_df.columns and not active_df[col].dropna().empty:
            method = str(active_df[col].dropna().iloc[0])
            break

    if method:
        label_text = f"Active {active_lens_name} Lens: <b>{method}</b> &nbsp;|&nbsp; Active Solutions: <b>{count}</b>"
    else:
        label_text = f"Active {active_lens_name} Lens &nbsp;|&nbsp; Active Solutions: <b>{count}</b>"

    st.markdown(
        f"""
        <div style="
            font-size: 17px; 
            font-weight: 500; 
            color: #1F2937; 
            background-color: #F3F4F6; 
            padding: 8px 14px; 
            border-radius: 6px; 
            border-left: 4px solid #3B82F6;
            margin-bottom: 16px;
            display: inline-block;
        ">
            {label_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================
# EXTRACTOR DE REGISTRO SOI & CONTROLES DE CONSENSUS
# =====================================================

def _unpack_registry(registry_obj: Any) -> List[Any]:
    """Extrae la lista/diccionario de SOIs desde cualquier tipo de registro."""
    if registry_obj is None:
        return []

    # 1. Si es un método o función
    if callable(registry_obj):
        try:
            registry_obj = registry_obj()
        except Exception:
            return []

    # 2. Si tiene métodos habituales de obtención de datos
    for method_name in ["get_all", "all", "get_sois", "list_sois", "to_dict", "values"]:
        if hasattr(registry_obj, method_name):
            attr = getattr(registry_obj, method_name)
            if callable(attr):
                try:
                    res = attr()
                    if res:
                        return list(res.values()) if isinstance(res, dict) else list(res)
                except Exception:
                    pass

    # 3. Si tiene atributos donde almacena la colección interna
    for attr_name in ["sois", "registry", "_registry", "_sois", "items", "data"]:
        if hasattr(registry_obj, attr_name):
            val = getattr(registry_obj, attr_name)
            if isinstance(val, dict):
                return list(val.values())
            elif isinstance(val, (list, tuple, set)):
                return list(val)

    # 4. Si el objeto de registro es en sí un diccionario
    if isinstance(registry_obj, dict):
        return list(registry_obj.values())

    # 5. Si el objeto es una lista o iterador
    if isinstance(registry_obj, (list, tuple, set)):
        return list(registry_obj)

    return []


def _parse_soi_item(item: Any) -> Dict[str, Any]:
    """Convierte un SOI individual (sea dict u objeto) a un formato estándar {'name', 'ids'}."""
    name = "Unnamed SOI"
    ids = []

    if isinstance(item, dict):
        name = str(item.get("name", item.get("label", item.get("title", item.get("id", "Unnamed SOI")))))
        raw_ids = item.get("ids", item.get("solution_ids", item.get("selected_ids", item.get("df", []))))
    else:
        name = str(getattr(item, "name", getattr(item, "label", getattr(item, "id", "Unnamed SOI"))))
        raw_ids = getattr(item, "ids", getattr(item, "solution_ids", getattr(item, "df", [])))

    if callable(raw_ids):
        try:
            raw_ids = raw_ids()
        except Exception:
            raw_ids = []

    if hasattr(raw_ids, "index") and not isinstance(raw_ids, (list, tuple, set)):
        # Si es un DataFrame o Series
        if hasattr(raw_ids, "columns") and "id" in raw_ids.columns:
            ids = raw_ids["id"].dropna().tolist()
        else:
            ids = raw_ids.index.tolist()
    elif isinstance(raw_ids, (list, tuple, set)):
        ids = list(raw_ids)

    return {"name": name, "ids": ids, "raw": item}


def _fetch_all_saved_sois(engine: Any = None) -> List[Dict[str, Any]]:
    """Escanea el engine.soi_registry y session_state recuperando todos los conjuntos guardados."""
    raw_sois = []

    # 1. Búsqueda prioritaria en engine.soi_registry
    if engine is not None and hasattr(engine, "soi_registry"):
        extracted = _unpack_registry(engine.soi_registry)
        if extracted:
            raw_sois.extend(extracted)

    # 2. Búsqueda secundaria en st.session_state si engine no devolvió nada
    if not raw_sois:
        for k in ["saved_sois", "sois", "saved_sets", "soi_list", "user_sois", "workspace_sois", "soi_registry"]:
            if k in st.session_state:
                extracted = _unpack_registry(st.session_state[k])
                if extracted:
                    raw_sois.extend(extracted)
                    break

    # 3. Normalizar todos los objetos SOI a {'name': ..., 'ids': ...}
    normalized = [_parse_soi_item(item) for item in raw_sois if item is not None]
    return normalized


def render_consensus_controls(engine: Any = None) -> Dict[str, Any]:
    """Renderiza controles de consensus detectando correctamente las SOIs guardadas."""
    saved_sois = _fetch_all_saved_sois(engine)

    if len(saved_sois) < 2:
        st.info("💡 Guarda al menos **2 SOIs** para poder combinarlas con Consensus.")

        # Desplegable de inspección por si el registro usa un nombre de campo distinto
        with st.expander("🔍 Inspeccionar contenido de soi_registry"):
            if engine is not None and hasattr(engine, "soi_registry"):
                reg = engine.soi_registry
                st.write("**Tipo de soi_registry:**", type(reg))
                st.write("**Atributos/Métodos:**", [a for a in dir(reg) if not a.startswith("__")])
                if hasattr(reg, "__dict__"):
                    st.write("**`__dict__` interno:**", reg.__dict__)
            else:
                st.write("No se encontró `soi_registry` en `engine`.")

        return {
            "method": "Consensus Threshold",
            "selected_sois": [],
            "target_sois": [],
            "threshold": 0.5,
        }

    soi_names = [soi["name"] for soi in saved_sois]

    method = st.selectbox(
        "Consensus Method",
        ["Consensus Threshold", "Union", "Majority", "Intersection"],
        key="consensus_method",
    )

    selected_names = st.multiselect(
        "SOIs to Combine",
        options=soi_names,
        default=soi_names[: min(2, len(soi_names))],
        key="consensus_selected_sois",
    )

    n_selected = len(selected_names)
    target_sois = [s for s in saved_sois if s["name"] in selected_names]

    params: Dict[str, Any] = {
        "method": method,
        "selected_sois": selected_names,
        "target_sois": target_sois,
    }

    if method == "Union":
        params["threshold"] = 1.0 / max(n_selected, 1)
        st.caption("Union keeps solutions supported by at least one selected SOI.")

    elif method == "Majority":
        params["threshold"] = 0.5
        st.caption("Majority keeps solutions supported by at least half of the selected SOIs.")

    elif method == "Intersection":
        params["threshold"] = 1.0
        st.caption("Intersection keeps only solutions supported by every selected SOI.")

    else:
        params["threshold"] = st.slider(
            "Consensus Level",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="consensus_threshold",
        )

        if params["threshold"] >= 0.75:
            st.caption("Mode: consensus core.")
        elif params["threshold"] >= 0.50:
            st.caption("Mode: consensus pool.")
        else:
            st.caption("Mode: broad exploratory pool.")

    st.caption(
        "This lens treats saved SOIs as expert opinions and combines them into one consensus SOI."
    )

    return params