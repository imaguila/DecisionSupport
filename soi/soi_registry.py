## --------------------------------------------------------------------------------------
## soi/soi_registry.py
## --------------------------------------------------------------------------------------

import html
import streamlit as st


# =====================================================
# SESSION STATE
# =====================================================

def ensure_soi_state():

    if "saved_sois" not in st.session_state:
        st.session_state.saved_sois = []


def has_loaded_soi():

    return (
        "active_soi_name"
        in st.session_state
        and
        "active_soi_ids"
        in st.session_state
    )


def clear_loaded_soi():

    if "active_soi_ids" in st.session_state:

        del st.session_state[
            "active_soi_ids"
        ]

    if "active_soi_name" in st.session_state:

        del st.session_state[
            "active_soi_name"
        ]

    if "active_soi_metadata" in st.session_state:

        del st.session_state[
            "active_soi_metadata"
        ]

    st.session_state[
        "pending_lens_reset"
    ] = True


def load_soi(
    soi
):

    st.session_state[
        "active_soi_ids"
    ] = soi[
        "ids"
    ]

    st.session_state[
        "active_soi_name"
    ] = soi[
        "name"
    ]

    st.session_state[
        "active_soi_metadata"
    ] = {
        "lens": soi.get(
            "lens"
        ),
        "method": soi.get(
            "method"
        ),
        "group": soi.get(
            "group"
        ),
        "group_column": soi.get(
            "group_column"
        ),
        "source_size": soi.get(
            "source_size"
        ),
        "soi_size": soi.get(
            "soi_size"
        ),
        "created_at": soi.get(
            "created_at"
        ),
        "params": soi.get(
            "params",
            {}
        )
    }

    st.session_state[
        "pending_lens_reset"
    ] = True


def delete_soi(
    idx
):

    deleted_name = (
        st.session_state
        .saved_sois[idx]["name"]
    )

    st.session_state.saved_sois.pop(
        idx
    )

    if (
        st.session_state.get(
            "active_soi_name"
        )
        ==
        deleted_name
    ):

        clear_loaded_soi()


# =====================================================
# LABEL HELPERS
# =====================================================

def normalize_method_label(
    soi
):

    method = soi.get(
        "method"
    )

    if method is None or method == "None":

        lens = soi.get(
            "lens",
            "Unknown"
        )

        if lens == "Exploratory":

            return "Exploratory"

        return None

    return method


def is_informative_group(
    group
):

    return (
        group is not None
        and group != ""
        and group != "All groups"
    )


def build_soi_main_label(
    soi
):

    name = soi.get(
        "name",
        "Unnamed SOI"
    )

    size = len(
        soi.get(
            "ids",
            []
        )
    )

    lens = soi.get(
        "lens",
        "Unknown"
    )

    method = normalize_method_label(
        soi
    )

    if method:

        return (
            f"{name} "
            f"[{size}] · "
            f"{lens} / {method}"
        )

    return (
        f"{name} "
        f"[{size}] · "
        f"{lens}"
    )


def build_compact_trace_label(
    soi
):

    parts = []

    group = soi.get(
        "group"
    )

    if is_informative_group(
        group
    ):

        parts.append(
            f"Group: {group}"
        )

    source_size = soi.get(
        "source_size"
    )

    soi_size = soi.get(
        "soi_size",
        len(
            soi.get(
                "ids",
                []
            )
        )
    )

    if source_size is not None:

        parts.append(
            f"{soi_size}/{source_size} solutions"
        )

    else:

        parts.append(
            f"{soi_size} solutions"
        )

    created_at = soi.get(
        "created_at"
    )

    if created_at:

        parts.append(
            created_at
        )

    return " · ".join(
        parts
    )


def build_tooltip_text(
    soi
):

    lens = soi.get(
        "lens",
        "Unknown"
    )

    method = normalize_method_label(
        soi
    )

    group = soi.get(
        "group"
    )

    source_size = soi.get(
        "source_size"
    )

    soi_size = soi.get(
        "soi_size",
        len(
            soi.get(
                "ids",
                []
            )
        )
    )

    created_at = soi.get(
        "created_at"
    )

    params = soi.get(
        "params",
        {}
    )

    lines = [
        f"Lens: {lens}",
        f"Method: {method if method else 'N/A'}",
        f"SOI size: {soi_size}"
    ]

    if source_size is not None:

        lines.append(
            f"Source size: {source_size}"
        )

    if is_informative_group(
        group
    ):

        lines.append(
            f"Group: {group}"
        )

    if created_at:

        lines.append(
            f"Created: {created_at}"
        )

    if params:

        compact_params = ", ".join(
            [
                f"{key}={value}"
                for key, value in params.items()
                if key not in [
                    "selected_sois",
                    "params"
                ]
            ]
        )

        if compact_params:

            lines.append(
                f"Params: {compact_params}"
            )

    return "\n".join(
        lines
    )


def render_trace_tooltip(
    soi
):

    compact_label = build_compact_trace_label(
        soi
    )

    tooltip = build_tooltip_text(
        soi
    )

    safe_label = html.escape(
        compact_label
    )

    safe_tooltip = html.escape(
        tooltip
    )

    st.markdown(
        (
            "<span "
            f"title=\"{safe_tooltip}\" "
            "style=\""
            "font-size:0.82rem;"
            "color:#6b7280;"
            "line-height:1.2;"
            "cursor:help;"
            "\">"
            f"ⓘ {safe_label}"
            "</span>"
        ),
        unsafe_allow_html=True
    )


# =====================================================
# RENDER LOADED SOI
# =====================================================

def render_loaded_soi_status():
    if not has_loaded_soi():

        return

    st.success(
        f"Active SOI: "
        f"{st.session_state.active_soi_name} "
        f"({len(st.session_state.active_soi_ids)} solutions)"
    )

    metadata = st.session_state.get(
        "active_soi_metadata",
        {}
    )

    if metadata:

        lens = metadata.get(
            "lens"
        )

        method = metadata.get(
            "method"
        )

        group = metadata.get(
            "group"
        )

        label_parts = []

        if lens:

            label_parts.append(
                lens
            )

        if method:

            label_parts.append(
                method
            )

        if is_informative_group(
            group
        ):

            label_parts.append(
                group
            )

        if label_parts:

            st.caption(
                " · ".join(
                    label_parts
                )
            )

    if st.button(
        "Clear Loaded SOI",
        use_container_width=True,
        key="clear_loaded_soi"
    ):

        clear_loaded_soi()

        st.rerun()

    st.markdown(
        "---"
    )


# =====================================================
# RENDER SAVED SOI ROW
# =====================================================
def render_saved_soi_row(
    soi,
    idx
):

    col_info, col_help, col_load, col_delete = st.columns(
        [
            0.68,
            0.08,
            0.14,
            0.10
        ]
    )

    with col_info:

        st.caption(
            f"• {build_soi_main_label(soi)}"
        )

    with col_help:

        tooltip = build_tooltip_text(
            soi
        )

        if st.button(
            "ⓘ",
            key=f"info_soi_{idx}",
            help=tooltip,
            use_container_width=True
        ):

            pass

    with col_load:

        if st.button(
            "Load",
            key=f"load_soi_{idx}",
            use_container_width=True
        ):

            load_soi(
                soi
            )

            st.rerun()

    with col_delete:

        if st.button(
            "🗑️",
            key=f"delete_soi_{idx}",
            use_container_width=True
        ):

            delete_soi(
                idx
            )

            st.rerun()


# =====================================================
# MAIN RENDERER
# =====================================================


def render_soi_tab():
    ensure_soi_state()
    if not st.session_state.saved_sois:

            st.info(
                "No saved SOIs."
            )

            return

    render_loaded_soi_status()

    for idx, soi in enumerate(
            st.session_state.saved_sois
    ):

        render_saved_soi_row(
                soi,
                idx
        )