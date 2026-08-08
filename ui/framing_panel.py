"""
Framing Panel UI Component.

Streamlit view component for interactive bounding range filters.
"""

from typing import Dict, Tuple
import streamlit as st
from core.pipeline import ParetoExplorerEngine
from core.framing import get_dimension_bounds
from ui.phase_help import render_phase_help_icon


def render_framing_summary(total_count: int, remaining_count: int) -> None:
    """Renders progress bar and ratio metrics summarizing solution reduction."""
    ratio = remaining_count / max(total_count, 1)
    st.progress(ratio)

    st.markdown(
        f"""
        <div style="text-align:center">
            <div style="font-size:0.85rem;color:gray;">
                Visible Decision Space
            </div>
            <div style="font-size:1.6rem;font-weight:bold;">
                {remaining_count} / {total_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"**{ratio:.1%}** of solutions satisfy the framing bounds.")


def render_framing_panel(engine: ParetoExplorerEngine) -> Dict[str, Tuple[float, float]]:
    """Genera sliders para todas las dimensiones (objetivos + indicadores enriquecidos)."""
    if not engine.has_data:
        st.info("Load a dataset to enable context framing.")
        return {}

    source_df = engine.working_df
    dimensions = engine.framing_dimensions
    available_bounds = get_dimension_bounds(source_df, dimensions)

    if not available_bounds:
        st.warning("No numeric dimensions available for framing.")
        return {}

    selected_bounds: Dict[str, Tuple[float, float]] = {}

    col_label, col_help = st.columns([0.85, 0.15], vertical_alignment="center")
    with col_label:
        st.markdown("**Bounded Range Filters**")
    with col_help:
        render_phase_help_icon("framing", key="help_framing_phase")

    for metric, (min_v, max_v) in available_bounds.items():
        step_val = max((max_v - min_v) / 1000.0, 0.001)

        selected_range = st.slider(
            metric,
            min_value=min_v,
            max_value=max_v,
            value=(min_v, max_v),
            step=step_val,
            key=f"framing_slider_{metric}",
        )

        is_modified = (
            abs(selected_range[0] - min_v) > 1e-5
            or abs(selected_range[1] - max_v) > 1e-5
        )
        if is_modified:
            selected_bounds[metric] = selected_range

    return selected_bounds

