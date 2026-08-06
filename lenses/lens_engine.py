## --------------------------------------------------------------------------------------
## lens_engine.py
## --------------------------------------------------------------------------------------

from lenses.lens_registry import (
    get_lens_module
)


def apply_lens(
    df,
    lens_name,
    params,
    dataset
):

    if df is None:

        return df

    if lens_name == "None":

        return df.copy()

    lens_module = get_lens_module(
        lens_name
    )

    if lens_module is None:

        return df.copy()

    return lens_module.apply(
        df,
        params,
        dataset
    )