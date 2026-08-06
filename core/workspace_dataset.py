## --------------------------------------------------------------------------------------
## core/workspace_dataset.py
## --------------------------------------------------------------------------------------

import streamlit as st

def get_ordered_columns( df, dataset ):

    var_prefix = (
        dataset["config"] .get( "var_prefix", "x_" )
    )

    objective_cols = ( dataset["metrics"] )
    indicator_cols = ( dataset["selected_indicators"] )
    decision_cols = [ col for col in df.columns
        if col.startswith(
            var_prefix ) ]

    control_cols = [ "highlight", "highlight_label", "label" ]

    other_cols = [
        col
        for col in df.columns
        if (
            col not in objective_cols
            and col not in indicator_cols
            and col not in decision_cols
            and col not in control_cols
            and col != "id"
        )
    ]

    ordered_cols = (
        ["id"] + objective_cols + indicator_cols +
        other_cols + decision_cols )

    ordered_cols = [ col for col in ordered_cols
        if col in df.columns ]

    return ordered_cols


def get_current_set_label():
    if st.session_state.get( "css_enabled", False ):
        return "Current CSS"
    return "Current Decision Set"


def render_dataset_table( df, dataset ):
    label = get_current_set_label()

    st.markdown( f"#### 📋 {label}" )
    ordered_cols = get_ordered_columns( df, dataset )

    st.dataframe(
        df[ordered_cols],
        use_container_width=True,
        height=420,
        hide_index=True
    )


def render_dataset_preview( df, dataset ):
    label = get_current_set_label()

    with st.expander(
        f"📋 {label} "
        f"(prefix: "
        f"{dataset['config'].get('var_prefix')})",
        expanded=False
    ):

        render_dataset_table( df, dataset )