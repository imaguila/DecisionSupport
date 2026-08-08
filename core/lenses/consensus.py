"""
Consensus Lens Module.

Aggregates multiple saved Sets of Interest (SOIs) into a unified consensus 
model using threshold-based voting logic, unions, majorities, or intersections.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st


# =====================================================
# EXTRACTORES DEFENSIVOS DE NOMBRES E IDs
# =====================================================

def _extract_soi_name(soi: Any) -> str:
    """Extrae el nombre de cualquier objeto o diccionario SOI sin fallar."""
    if soi is None:
        return "Unnamed SOI"

    # Si es una tupla/lista tipo ("Nombre", objeto_soi)
    if isinstance(soi, (tuple, list)) and len(soi) > 0:
        if isinstance(soi[0], str):
            return soi[0]
        soi = soi[-1]

    # Si es un diccionario
    if isinstance(soi, dict):
        for key in ["name", "label", "title", "id", "soi_name"]:
            if key in soi and soi[key]:
                return str(soi[key])
        if "raw" in soi and soi["raw"] is not soi:
            return _extract_soi_name(soi["raw"])
        return "Unnamed SOI"

    # Si es un objeto de clase personalizada (e.g. SOI)
    for attr in ["name", "label", "title", "id", "soi_name"]:
        if hasattr(soi, attr):
            val = getattr(soi, attr, None)
            if val is not None and not callable(val):
                return str(val)
            elif callable(val):
                try:
                    res = val()
                    if res:
                        return str(res)
                except Exception:
                    pass

    # Inspeccionar __dict__ interno si existe
    if hasattr(soi, "__dict__"):
        d = soi.__dict__
        for k in ["name", "_name", "label", "_label"]:
            if k in d and d[k]:
                return str(d[k])

    return "Unnamed SOI"


def _extract_soi_ids(soi: Any) -> List[Any]:
    """Extrae la lista de IDs de cualquier objeto o diccionario SOI sin fallar."""
    if soi is None:
        return []

    # Desempaquetar tuplas si aplica
    if isinstance(soi, (tuple, list)) and len(soi) > 0:
        if not isinstance(soi[0], (int, str)):
            for item in soi:
                ids = _extract_soi_ids(item)
                if ids:
                    return ids

    raw_ids = None

    # Desempaquetar 'raw' si es un envoltorio
    if isinstance(soi, dict) and "raw" in soi:
        if "ids" in soi and soi["ids"]:
            raw_ids = soi["ids"]
        else:
            return _extract_soi_ids(soi["raw"])

    if raw_ids is None:
        # Búsqueda en Diccionarios
        if isinstance(soi, dict):
            for k in ["ids", "solution_ids", "selected_ids", "solutions", "df", "data"]:
                if k in soi and soi[k] is not None:
                    raw_ids = soi[k]
                    break
        else:
            # Búsqueda en Atributos/Métodos de objeto
            for attr in ["ids", "solution_ids", "selected_ids", "solutions", "df", "data"]:
                if hasattr(soi, attr):
                    val = getattr(soi, attr, None)
                    if val is not None:
                        raw_ids = val
                        break

    if callable(raw_ids):
        try:
            raw_ids = raw_ids()
        except Exception:
            raw_ids = []

    if raw_ids is None:
        return []

    # Si se obtuvo un DataFrame o Series
    if isinstance(raw_ids, pd.DataFrame):
        id_col = _find_id_column(raw_ids)
        if id_col:
            return raw_ids[id_col].dropna().tolist()
        return raw_ids.index.dropna().tolist()
    elif isinstance(raw_ids, pd.Series):
        return raw_ids.dropna().tolist()

    # Iterable estándar
    if hasattr(raw_ids, "__iter__") and not isinstance(raw_ids, (str, bytes, dict)):
        return list(raw_ids)

    return []


def _to_clean_str(val: Any) -> str:
    """Normaliza IDs para evitar fallos de coincidencia entre int, float y str."""
    if pd.isna(val):
        return ""
    try:
        fval = float(val)
        if fval.is_integer():
            return str(int(fval))
    except (ValueError, TypeError):
        pass
    return str(val).strip()


def _find_id_column(df: pd.DataFrame) -> str:
    """Detecta el nombre de la columna identificadora de soluciones."""
    for col in ["id", "sol_id", "solution_id", "ID", "Sol_ID"]:
        if col in df.columns:
            return col
    return ""


def _get_selected_sois(
    selected_names: List[str], target_sois_param: Optional[List[Any]] = None
) -> List[Any]:
    """Obtiene y filtra la lista de SOIs seleccionadas."""
    candidate_sois: List[Any] = []

    if target_sois_param:
        candidate_sois = target_sois_param
    else:
        saved_sois: List[Any] = st.session_state.get("saved_sois", [])
        if not saved_sois and "sois" in st.session_state:
            raw = st.session_state["sois"]
            saved_sois = list(raw.values()) if isinstance(raw, dict) else list(raw)
        candidate_sois = saved_sois

    # Filtrar preservando coincidencias de nombre
    matched = [soi for soi in candidate_sois if _extract_soi_name(soi) in selected_names]
    return matched if matched else candidate_sois


# =====================================================
# CÁLCULO DE CONSENSO Y SOPORTE
# =====================================================

def _build_support_table(selected_sois: List[Any]) -> pd.DataFrame:
    """Calcula el número de votos (soporte) de cada solución en las SOIs."""
    support: Dict[Any, int] = {}
    support_names: Dict[Any, List[str]] = {}

    for soi in selected_sois:
        soi_name = _extract_soi_name(soi)
        unique_ids = set(_extract_soi_ids(soi))

        for solution_id in unique_ids:
            support[solution_id] = support.get(solution_id, 0) + 1
            support_names.setdefault(solution_id, []).append(soi_name)

    rows = []
    n_sois = len(selected_sois)

    for solution_id, support_count in support.items():
        consensus_score = support_count / max(n_sois, 1)
        rows.append(
            {
                "id": solution_id,
                "consensus_support_count": support_count,
                "consensus_score": consensus_score,
                "consensus_supporting_sois": ", ".join(
                    sorted(support_names.get(solution_id, []))
                ),
            }
        )

    return pd.DataFrame(rows)


def _add_consensus_labels(result: pd.DataFrame, n_sois: int) -> pd.DataFrame:
    """Añade etiquetas informativas de soporte al DataFrame de resultado."""
    result["group_base"] = result["consensus_support_count"].apply(
        lambda count: f"Support = {int(count)}/{n_sois}"
    )

    group_sizes = result["group_base"].value_counts().to_dict()

    result["group_label"] = result["group_base"].apply(
        lambda grp: f"{grp} (n={group_sizes.get(grp, 0)})"
    )

    return result


# =====================================================
# MÉTODOS PRINCIPALES DE LA LENTE
# =====================================================

def apply(
    df: pd.DataFrame,
    params: Dict[str, Any],
    dataset: Optional[Any] = None,
    context: Optional[Any] = None,
) -> pd.DataFrame:
    """Aplica el cálculo de consenso al DataFrame activo."""
    if df is None or df.empty:
        return df

    result = df.copy()
    selected_names = params.get("selected_sois", [])
    target_sois_param = params.get("target_sois")

    selected_sois = _get_selected_sois(selected_names, target_sois_param)

    if len(selected_sois) < 2:
        return result

    support_table = _build_support_table(selected_sois)

    if support_table.empty:
        return result.iloc[0:0].copy()

    threshold = params.get("threshold", 0.5)
    support_table = support_table[
        support_table["consensus_score"] >= (threshold - 1e-9)
    ].copy()

    if support_table.empty:
        return result.iloc[0:0].copy()

    # Identificar columna clave para la unión
    id_col = _find_id_column(result)

    if id_col and id_col != "id":
        support_table = support_table.rename(columns={"id": id_col})
        merge_col = id_col
    elif id_col == "id":
        merge_col = "id"
    else:
        result["_temp_idx"] = result.index
        support_table = support_table.rename(columns={"id": "_temp_idx"})
        merge_col = "_temp_idx"

    # Garantizar compatibilidad exacta de tipos de ID
    result["_merge_key"] = result[merge_col].apply(_to_clean_str)
    support_table["_merge_key"] = support_table[merge_col].apply(_to_clean_str)

    result = result.merge(
        support_table.drop(columns=[merge_col]), on="_merge_key", how="inner"
    ).drop(columns=["_merge_key"])

    if "_temp_idx" in result.columns:
        result = result.drop(columns=["_temp_idx"])

    n_sois = len(selected_sois)
    result = _add_consensus_labels(result, n_sois)
    result["consensus_method"] = params.get("method", "Consensus Threshold")
    result["consensus_threshold"] = threshold
    result["consensus_source_sois"] = ", ".join(selected_names)

    sort_cols = ["consensus_score", "consensus_support_count"]
    if merge_col in result.columns:
        sort_cols.append(merge_col)

    result = result.sort_values(sort_cols, ascending=[False, False, True]).copy()
    result["consensus_rank"] = range(1, len(result) + 1)

    return result


def render_feedback(lens_df: Optional[pd.DataFrame]) -> None:
    """Muestra métricas contextuales en la interfaz."""
    if lens_df is None or lens_df.empty:
        st.warning("The consensus SOI is empty or not available.")
        return

    if "consensus_method" in lens_df.columns:
        method = lens_df["consensus_method"].iloc[0]
        st.info(f"Consensus method: {method}")

    if "consensus_threshold" in lens_df.columns:
        threshold = lens_df["consensus_threshold"].iloc[0]
        st.caption(f"Consensus threshold: {float(threshold):.2f}")

    if "consensus_score" in lens_df.columns:
        max_score = lens_df["consensus_score"].max()
        st.caption(f"Maximum consensus score: {float(max_score):.2f}")

    st.caption(f"Consensus SOI size: {len(lens_df)} solutions")