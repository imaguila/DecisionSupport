## --------------------------------------------------------------------------------------
## ui/visualization.py
## --------------------------------------------------------------------------------------

import plotly.express as px
import streamlit as st
import pandas as pd

# =====================================================
# COLOR SELECTION
# =====================================================

def infer_lens_color_column( df, user_color=None ):

    # --------------------------------------------------
    # Priority order:
    # 1. Group labels from clustering / indicator dominance
    # 2. Cluster labels
    # 3. Preference score
    # 4. Efficiency score
    # 5. Indicator dominance score
    # 6. User-selected color
    # --------------------------------------------------

    if "group_label" in df.columns:
        return "group_label"

    if "cluster_str" in df.columns:
        return "cluster_str"

    if "preference_score" in df.columns:
        return "preference_score"

    if "efficiency_score" in df.columns:
        return "efficiency_score"

    if "consensus_score" in df.columns:
        return "consensus_score"

    if "domain_match_count" in df.columns:
        return "domain_match_count"

    return user_color


def is_discrete_color( df, color_column ):

    if color_column is None:
        return False

    if color_column not in df.columns:
        return False

    if color_column in [
        "group_label",
        "cluster_str",
        "preference_method",
        "efficiency_method",
        "domain_matched_metrics"
    ]:

        return True

    if pd.api.types.is_object_dtype( df[color_column] ):
        return True

    return False


def build_hover_columns( df ):

    excluded_prefixes = ( "req_", "var_", "x_" )

    excluded_cols = {
        "label",
        "highlight",
        "highlight_label"
    }

    hover_cols = []

    for col in df.columns:

        if col in excluded_cols:
            continue

        if col.startswith( excluded_prefixes ):
            continue

        hover_cols.append( col)

    return hover_cols


# =====================================================
# SCATTER
# =====================================================

def render_scatter(
    df,
    x,
    y,
    size=None,
    color=None,
    show_ids=False,
    key=None
):

    df = df.copy()

    if x not in df.columns or y not in df.columns:
        st.warning( "Selected axes are not available in the current dataset." )
        return

    text_column = None

    if show_ids:

        if "id" in df.columns:
            text_column = "id"

        elif "ID" in df.columns:
            text_column = "ID"

    plot_color = infer_lens_color_column(
        df,
        user_color=color
    )

    discrete_color = is_discrete_color(
        df,
        plot_color
    )

    hover_cols = build_hover_columns(
        df
    )

    if discrete_color and plot_color is not None:

        df[
            plot_color
        ] = df[
            plot_color
        ].astype(
            str
        )

        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=plot_color,
            text=text_column,
            hover_data=hover_cols
        )

    else:

        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=plot_color,
            color_continuous_scale=px.colors.sequential.Viridis,
            text=text_column,
            hover_data=hover_cols
        )

    fig.update_traces(
        textposition="top center",
        textfont=dict(
            size=10
        )
    )

    if (
        "highlight"
        in df.columns
        and df["highlight"].any()
    ):

        marker_opacity = df[
            "highlight"
        ].apply(
            lambda value: 1.0 if value else 0.25
        )

        fig.update_traces(
            marker=dict(
                opacity=marker_opacity
            )
        )

        

    fig.update_layout(
        height=500,
        template="plotly_white",
        legend_title_text=(
            plot_color
            if plot_color is not None
            else ""
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key
    )


# =====================================================
# COORDINATED MAPS
# =====================================================

def render_coordinated_maps(
    df,
    x,
    y,
    z,
    key_prefix,
    show_ids=False
):

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.caption(
            f"{x} vs {y}"
        )

        render_scatter(
            df,
            x=x,
            y=y,
            show_ids=show_ids,
            key=f"{key_prefix}_left"
        )

    with col2:

        st.caption(
            f"{x} vs {z}"
        )

        render_scatter(
            df,
            x=x,
            y=z,
            show_ids=show_ids,
            key=f"{key_prefix}_right"
        )


# =====================================================
# DISTRIBUTION
# =====================================================

def render_distribution(
    df,
    metric,
    mode="Violin",
    key=None
):

    if metric not in df.columns:

        st.warning(
            "Selected metric is not available in the current dataset."
        )

        return

    if mode == "Violin":

        fig = px.violin(
            df,
            y=metric,
            box=True,
            points="all"
        )

    else:

        fig = px.box(
            df,
            y=metric,
            points="all"
        )

    fig.update_layout(
        title=f"Distribution of {metric}",
        height=550,
        showlegend=False,
        template="plotly_white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key
    )