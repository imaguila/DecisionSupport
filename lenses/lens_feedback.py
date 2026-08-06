## --------------------------------------------------------------------------------------
## lens_feedback.py
## --------------------------------------------------------------------------------------


from lenses.lens_registry import (
    get_lens_module
)


def render_lens_feedback(
    placeholder,
    active_lens,
    lens_df
):

    if placeholder is None:
        return

    if lens_df is None:

        return

    if active_lens == "None":

        return

    lens_module = get_lens_module(
        active_lens
    )

    if lens_module is None:

        return

    with placeholder.container():

        if hasattr(
            lens_module,
            "render_feedback"
        ):

            lens_module.render_feedback(
                lens_df
            )