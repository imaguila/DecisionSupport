import streamlit as st
from core.pipeline import ParetoExplorerEngine
from ui.input_panel import render_input_panel
from ui.enrichment_panel import render_enrichment_panel
from ui.framing_panel import render_framing_panel, render_framing_summary
from ui.lens_panel import render_lens_panel, render_lens_feedback
from ui.workspace_summary import render_summary
from ui.workspace_control import render_workspace_controls
from ui.workspace_maps import render_maps
from ui.css_panel import render_css_panel, render_css_comparison

st.set_page_config(page_title="Decision Space Explorer", layout="wide")
st.title("Decision Space Explorer")

if "engine" not in st.session_state:
    st.session_state.engine = ParetoExplorerEngine()

engine: ParetoExplorerEngine = st.session_state.engine

active_lens_name = "None"
lens_params = {}
css_df = None

# --------------------------------------------------------------------------------------
# Sidebar Workflow
# --------------------------------------------------------------------------------------
with st.sidebar:
    # Phase 1: Input
    problem_context = render_input_panel()
    if problem_context is not None:
        engine.load_problem(problem_context)

    if engine.has_data and engine.plugin:
        with st.expander("⚙️ Data Enrichment", expanded=False):
            selected_indicators = render_enrichment_panel(engine)
            if st.button("Apply Enrichment", type="primary"):
                engine.enrich(selected_indicators)
                st.success("Dataset enriched successfully!")
                st.rerun()

    if engine.has_data:
        with st.expander("🎛️ Context Framing", expanded=False):
            user_bounds = render_framing_panel(engine)
            engine.apply_framing(user_bounds)
            render_framing_summary(len(engine.working_df), len(engine.framed_df))

    show_ids = False
    if engine.has_data:
        show_ids = render_workspace_controls(engine.framing_dimensions)

    if engine.has_data:
        active_lens_name, lens_params = render_lens_panel(engine)

    # 2. PANEL CSS EN EL SIDEBAR (Fija el SOI activo como CSS)
    if engine.has_data:
        # Construimos el diccionario de contexto del dataset
        dataset_config = getattr(engine.context, "config", {}) if engine.context else {}
        dataset_ctx = {
            "metrics": getattr(engine, "metrics", engine.framing_dimensions),
            "selected_indicators": getattr(engine, "selected_indicators", []),
            "config": dataset_config,
        }
        
        # Renderiza controles en sidebar y obtiene el DataFrame fijado/filtrado
        css_df = render_css_panel(current_df=engine.active_df, dataset=dataset_ctx)


# --------------------------------------------------------------------------------------
# Main Workspace View
# --------------------------------------------------------------------------------------
if engine.has_data:
    dataset_config = getattr(engine.context, "config", {}) if engine.context else {}
    dataset_ctx = {
        "metrics": getattr(engine, "metrics", engine.framing_dimensions),
        "selected_indicators": getattr(engine, "selected_indicators", []),
        "config": dataset_config,
    }

    # 1. Summary (Contiene Overview, Table Preview y la pestaña de Saved SOIs)
    render_summary(engine=engine, dataset_config=dataset_config)

    # Banner de información de lente activa
    render_lens_feedback(active_lens_name, engine.active_df)

    # 2. Decision Maps Section
    render_maps(
        df=engine.active_df,
        dimensions=engine.framing_dimensions,
        show_ids=show_ids,
    )

    # 3. DETAILED CSS COMPARISON SECTION
    # Se renderiza sólo si el checkbox "Open detailed comparison" está activo en el sidebar
    if css_df is not None:
        render_css_comparison(css_df=css_df, dataset=dataset_ctx)

else:
    st.info("Select a dataset in the sidebar to begin.")