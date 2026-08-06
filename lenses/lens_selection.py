## --------------------------------------------------------------------------------------
## lens_selection.py
## --------------------------------------------------------------------------------------

import streamlit as st


def ensure_soi_state():

    if "saved_sois" not in st.session_state:

        st.session_state.saved_sois = []


def get_group_column(
    lens_df
):

    if lens_df is None:

        return None

    if "group_label" in lens_df.columns:

        return "group_label"

    if "cluster_str" in lens_df.columns:

        return "cluster_str"

    return None


def get_group_options(
    lens_df,
    group_column
):

    if group_column is None:

        return []

    return sorted(
        lens_df[group_column]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


def filter_by_group(
    lens_df,
    group_column,
    group_value
):

    if lens_df is None:

        return None

    if group_column is None:

        return lens_df.copy()

    if group_value == "All groups":

        return lens_df.copy()

    return lens_df[
        lens_df[group_column].astype(str)
        ==
        str(group_value)
    ].copy()


def get_lens_label(
    active_lens
):

    if active_lens == "None":

        return "Exploratory"

    return active_lens


def reset_soi_name_if_needed(
    active_lens,
    group_value
):

    lens_label = get_lens_label(
        active_lens
    )

    suffix = (
        group_value
        if group_value != "All groups"
        else "Current set"
    )

    default_name = (
        f"{lens_label} - {suffix} "
        f"#{len(st.session_state.saved_sois) + 1}"
    )

    name_context = (
        lens_label,
        group_value
    )

    if (
        st.session_state.get(
            "soi_name_context"
        )
        != name_context
    ):

        st.session_state[
            "soi_name"
        ] = default_name

        st.session_state[
            "soi_name_context"
        ] = name_context


def render_group_selector_and_save(
    placeholder,
    active_lens,
    lens_df,
    lens_params
):

    ensure_soi_state()

    if lens_df is None:
        return lens_df

    with placeholder.container():
        lens_label = get_lens_label(
            active_lens
        )

        group_column = get_group_column(
            lens_df
        )

        group_value = "All groups"
        
        if group_column is not None:

            group_options = get_group_options(
                lens_df,
                group_column
            )

            options = (
                ["All groups"]
                +
                group_options
            )

            selector_key = (
                f"soi_group_selector_"
                f"{lens_label.replace(' ', '_')}"
            )

            if (
                st.session_state.get(
                    selector_key
                )
                not in options
            ):

                st.session_state[
                    selector_key
                ] = "All groups"

            group_value = st.selectbox(
                "SOI group",
                options,
                key=selector_key,
                help=(
                    "Choose the group to promote as the current "
                    "Solution of Interest."
                )
            )

        current_df = filter_by_group(
            lens_df,
            group_column,
            group_value
        )

        if current_df is None:

            return lens_df

        st.caption(
            f"Current SOI candidate size: "
            f"{len(current_df)} solutions"
        )

        if active_lens == "None":

            st.caption(
                "Source: exploratory current set."
            )

        st.markdown(
            "---"
        )

        reset_soi_name_if_needed(
            active_lens,
            group_value
        )

        soi_name = st.text_input(
            "Name",
            key="soi_name"
        )

        if st.button(
            "💾 Save Current Set",
            use_container_width=True,
            key="save_current_soi"
        ):

            st.session_state.pending_save_soi = {
                "name": soi_name,
                "lens": lens_label,
                "method": lens_params.get(
                    "method",
                    "Exploratory"
                ),
                "params": lens_params,
                "ids": current_df["id"].tolist(),
                "group": group_value,
                "group_column": group_column,
                "source_size": len(lens_df),
                "soi_size": len(current_df)
            }

        return current_df