"""
Sets of Interest (SOI) Registry Module.

Provides session state lifecycle management, formatting utilities, and 
Streamlit UI components for saving, inspecting, loading, and deleting candidate 
Sets of Interest (SOIs).
"""

import html
from typing import Any, Dict, List, Optional

import streamlit as st


# =====================================================
# SESSION STATE MANAGEMENT
# =====================================================


def ensure_soi_state() -> None:
    """Ensures session state contains required keys for SOI registry operations."""
    if "saved_sois" not in st.session_state:
        st.session_state.saved_sois = []


def has_loaded_soi() -> bool:
    """
    Checks whether an active SOI is currently loaded in session state.

    Returns
    -------
    bool
        True if both active SOI name and active SOI IDs exist in session state.
    """
    return (
        "active_soi_name" in st.session_state
        and "active_soi_ids" in st.session_state
    )


def clear_loaded_soi() -> None:
    """Clears the currently active SOI from session state and triggers lens reset."""
    for key in ["active_soi_ids", "active_soi_name", "active_soi_metadata"]:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["pending_lens_reset"] = True


def load_soi(soi: Dict[str, Any]) -> None:
    """
    Loads a saved SOI dictionary into active session state.

    Parameters
    ----------
    soi : Dict[str, Any]
        Dictionary representing the SOI configuration and solution IDs.
    """
    st.session_state["active_soi_ids"] = soi.get("ids", [])
    st.session_state["active_soi_name"] = soi.get("name", "Unnamed SOI")
    st.session_state["active_soi_metadata"] = {
        "lens": soi.get("lens"),
        "method": soi.get("method"),
        "group": soi.get("group"),
        "group_column": soi.get("group_column"),
        "source_size": soi.get("source_size"),
        "soi_size": soi.get("soi_size", len(soi.get("ids", []))),
        "created_at": soi.get("created_at"),
        "params": soi.get("params", {}),
    }

    st.session_state["pending_lens_reset"] = True


def delete_soi(idx: int) -> None:
    """
    Deletes an SOI entry from the registry by list index.

    Parameters
    ----------
    idx : int
        Index of the SOI to remove from `st.session_state.saved_sois`.
    """
    ensure_soi_state()
    saved_sois: List[Dict[str, Any]] = st.session_state.saved_sois

    if 0 <= idx < len(saved_sois):
        deleted_soi = saved_sois.pop(idx)
        deleted_name = deleted_soi.get("name")

        if st.session_state.get("active_soi_name") == deleted_name:
            clear_loaded_soi()


# =====================================================
# LABEL & FORMATTING HELPERS
# =====================================================


def normalize_method_label(soi: Dict[str, Any]) -> Optional[str]:
    """
    Normalizes the method label extracted from an SOI dictionary.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.

    Returns
    -------
    Optional[str]
        Normalized method label string or None if uninformative.
    """
    method = soi.get("method")

    if method is None or method == "None":
        lens = soi.get("lens", "Unknown")
        if lens == "Exploratory":
            return "Exploratory"
        return None

    return str(method)


def is_informative_group(group: Any) -> bool:
    """
    Determines if a group identifier represents a non-trivial subgroup filter.

    Parameters
    ----------
    group : Any
        Group name or filter value.

    Returns
    -------
    bool
        True if group is non-empty and distinct from default global indicators.
    """
    return (
        group is not None
        and group != ""
        and group != "All groups"
    )


def build_soi_main_label(soi: Dict[str, Any]) -> str:
    """
    Constructs the primary label string displayed in registry lists.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.

    Returns
    -------
    str
        Formatted main label string.
    """
    name = soi.get("name", "Unnamed SOI")
    size = len(soi.get("ids", []))
    lens = soi.get("lens", "Unknown")
    method = normalize_method_label(soi)

    if method:
        return f"{name} [{size}] · {lens} / {method}"

    return f"{name} [{size}] · {lens}"


def build_compact_trace_label(soi: Dict[str, Any]) -> str:
    """
    Constructs a compact single-line provenance trace label.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.

    Returns
    -------
    str
        Compact provenance trace summary string.
    """
    parts = []
    group = soi.get("group")

    if is_informative_group(group):
        parts.append(f"Group: {group}")

    source_size = soi.get("source_size")
    soi_size = soi.get("soi_size", len(soi.get("ids", [])))

    if source_size is not None:
        parts.append(f"{soi_size}/{source_size} solutions")
    else:
        parts.append(f"{soi_size} solutions")

    created_at = soi.get("created_at")
    if created_at:
        parts.append(str(created_at))

    return " · ".join(parts)


def build_tooltip_text(soi: Dict[str, Any]) -> str:
    """
    Generates plain text tooltip information describing complete SOI parameters.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.

    Returns
    -------
    str
        Multi-line plain text summary for tooltip rendering.
    """
    lens = soi.get("lens", "Unknown")
    method = normalize_method_label(soi)
    group = soi.get("group")
    source_size = soi.get("source_size")
    soi_size = soi.get("soi_size", len(soi.get("ids", [])))
    created_at = soi.get("created_at")
    params = soi.get("params", {})

    lines = [
        f"Lens: {lens}",
        f"Method: {method if method else 'N/A'}",
        f"SOI size: {soi_size}",
    ]

    if source_size is not None:
        lines.append(f"Source size: {source_size}")

    if is_informative_group(group):
        lines.append(f"Group: {group}")

    if created_at:
        lines.append(f"Created: {created_at}")

    if params and isinstance(params, dict):
        compact_params = ", ".join(
            [
                f"{k}={v}"
                for k, v in params.items()
                if k not in ["selected_sois", "params"]
            ]
        )
        if compact_params:
            lines.append(f"Params: {compact_params}")

    return "\n".join(lines)


def render_trace_tooltip(soi: Dict[str, Any]) -> None:
    """
    Renders an HTML-escaped inline provenance trace tag with hover tooltip.

    Parameters
    ----------
    soi : Dict[str, Any]
        SOI metadata dictionary.
    """
    compact_label = build_compact_trace_label(soi)
    tooltip = build_tooltip_text(soi)

    safe_label = html.escape(compact_label)
    safe_tooltip = html.escape(tooltip)

    st.markdown(
        f'<span title="{safe_tooltip}" style="'
        f'font-size:0.82rem; color:#6b7280; line-height:1.2; cursor:help;'
        f'">ⓘ {safe_label}</span>',
        unsafe_allow_html=True,
    )


# =====================================================
# UI RENDER COMPONENTS
# =====================================================


def render_loaded_soi_status() -> None:
    """Renders status header and control buttons for the currently active loaded SOI."""
    if not has_loaded_soi():
        return

    active_name = st.session_state.active_soi_name
    active_ids = st.session_state.active_soi_ids

    st.success(f"Active SOI: {active_name} ({len(active_ids)} solutions)")

    metadata = st.session_state.get("active_soi_metadata", {})
    if metadata:
        lens = metadata.get("lens")
        method = metadata.get("method")
        group = metadata.get("group")

        label_parts = []
        if lens:
            label_parts.append(str(lens))
        if method:
            label_parts.append(str(method))
        if is_informative_group(group):
            label_parts.append(str(group))

        if label_parts:
            st.caption(" · ".join(label_parts))

    if st.button(
        "Clear Loaded SOI",
        use_container_width=True,
        key="clear_loaded_soi",
    ):
        clear_loaded_soi()
        st.rerun()

    st.divider()


def render_saved_soi_row(soi: Dict[str, Any], idx: int) -> None:
    """
    Renders a single row entry for a saved SOI in the registry panel.

    Parameters
    ----------
    soi : Dict[str, Any]
        Target SOI metadata dictionary.
    idx : int
        Registry list index.
    """
    col_info, col_help, col_load, col_delete = st.columns([0.68, 0.08, 0.14, 0.10])

    with col_info:
        st.caption(f"• {build_soi_main_label(soi)}")

    with col_help:
        tooltip = build_tooltip_text(soi)
        st.button(
            "ⓘ",
            key=f"info_soi_{idx}",
            help=tooltip,
            use_container_width=True,
        )

    with col_load:
        if st.button(
            "Load",
            key=f"load_soi_{idx}",
            use_container_width=True,
        ):
            load_soi(soi)
            st.rerun()

    with col_delete:
        if st.button(
            "🗑️",
            key=f"delete_soi_{idx}",
            use_container_width=True,
        ):
            delete_soi(idx)
            st.rerun()


# =====================================================
# MAIN TAB RENDERER
# =====================================================


def render_soi_tab() -> None:
    """Renders the complete Sets of Interest (SOI) registry tab interface."""
    ensure_soi_state()

    if not st.session_state.saved_sois:
        st.info("No saved SOIs.")
        return

    render_loaded_soi_status()

    for idx, soi in enumerate(st.session_state.saved_sois):
        render_saved_soi_row(soi, idx)