## --------------------------------------------------------------------------------------
## lenses/lens_manual.py
## --------------------------------------------------------------------------------------

import streamlit as st


def render_params(
    dataset,
    working_df
):
    """
    Renderiza los controles del sidebar para elegir soluciones manualmente una por una.
    """
    params = {
        "method": "Manual Selection"
    }

    if working_df is None or working_df.empty or "id" not in working_df.columns:
        st.warning("No solutions available for manual selection.")
        params["selected_ids"] = []
        return params

    valid_ids = (
        working_df["id"]
        .dropna()
        .astype(int)
        .tolist()
    )

    params["selected_ids"] = st.multiselect(
        "Pick solutions one by one",
        options=valid_ids,
        default=[],
        key="manual_lens_selected_ids",
        help="Manually pick the exact solutions you want to isolate."
    )

    return params


def apply(
    df,
    params,
    dataset
):
    """
    Filtra el dataframe manteniendo únicamente los IDs seleccionados.
    """
    if df is None or df.empty:
        return df

    selected_ids = params.get("selected_ids", [])

    if not selected_ids:
        # Si no se ha seleccionado ninguna solución, retorna un dataframe vacío con la misma estructura
        return df.iloc[0:0].copy()

    return df[
        df["id"].isin(selected_ids)
    ].copy()


def render_feedback(
    lens_df
):
    """
    Muestra información de estado en la UI cuando esta lens está activa.
    """
    if lens_df is None:
        return

    count = len(lens_df)
    if count == 0:
        st.caption("No solutions selected in manual lens.")
    else:
        st.info(f"📌 Manual selection: {count} solution(s) active.")