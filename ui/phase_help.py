## --------------------------------------------------------------------------------------
## ui/phase_help.py
## --------------------------------------------------------------------------------------

import streamlit as st


# =====================================================
# PHASE HELP TEXTS
# =====================================================

PHASE_HELP = {
    "input": """
Load the base decision space.

**1. Domain Configuration**

Use this option when you want to load a predefined case already configured in the library.

A domain configuration usually provides:

- the solution dataset
- the default optimization objectives
- the decision-variable prefix
- optional plugin logic
- optional default indicators

This is the recommended option when the decision problem has a known structure.

**2. Upload Enriched CSV**

Use this option when you already have a standalone CSV.

The uploaded CSV should contain:

- one row per solution
- numeric objective or indicator columns
- decision-variable columns sharing a common prefix

Examples of decision-variable prefixes:

- `x_`
- `var_`
- `req_`
- `feature_`
- `design_`

After loading the data, refine the Objective Column (optimization objectives) that define the base decision space.

> **User Purpose:** Load the base Pareto-optimal alternatives to start transforming them into an interpretable decision space.

""",

    "enrichment": """
The goal of this stage is to **expand the descriptive layer of the solutions** by adding domain-specific quality and semantic indicators.

Available indicators are provided by the active domain plugin. The app checks
which indicators can be computed from the currently selected base objectives.

Only compatible indicators are shown. An indicator is compatible when all
required input columns are available in the current dataset.

Each indicator must also have its calculation logic defined in the plugin.
If the plugin does not define how an indicator is computed, the app cannot
generate that indicator.

Derived indicators enrich the decision space with additional analytical views,
such as:

- productivity
- scope
- quality
- efficiency
- domain-specific measures

> **User Purpose:** Uncover implicit properties in solutions to project, filter, and compare the decision space under richer analytical perspectives.


""",

    "framing": """
The goal of this stage is to **restrict the global decision space to current operational conditions**.

Use constraints in this section to delimit the active set by applying:
- Maximum effort (budget) thresholds.
- Risk limits or minimum productivity/scope requirements.

> **User Purpose:** Reduce analytical complexity by isolating only those solutions that are viable under the current strategic context before applying analytical lenses.
""",

    "workspace_controls": """
Create and manage decision-space maps.

Maps visualize the current decision set, SOI, or CSS using selected objectives and indicators.

The goal of Maps is to **project and visualize alternatives across multiple dimensions simultaneously**.

Configure the axes (X, Y) and marker size to inspect how objectives and derived indicators behave across the active solution set.

> **User Purpose:** Detect visual patterns, trade-offs, and spatial distributions without committing to a single ranking upfront.
""",

    "soi": """
Generate or load a Solution of Interest.

A SOI is a candidate subset of solutions. It can come from:

- an analytical lens
- a saved SOI
- a consensus of saved SOIs
- the exploratory current set
The goal of this stage is to **identify Solutions of Interest (SOIs)**: sub-sets with strategic coherence identified under a specific analytical perspective (Lens).

Rather than evaluating isolated solutions, apply different Analytical Lenses:
- **Preference (MCDA):** Discover top-N alternatives based on TOPSIS or Weighted Sum.
- **Diversity:** Group structurally representative solutions using clustering (K-Medoids, HDBSCAN).
- **Efficiency:** Identify solutions offering the best benefit-cost trade-offs.
- **Dominance:** Highlight alternatives that repeatedly excel across multiple quality indicators.

> **User Purpose:** Extract latent strategic insights and patterns from the decision space using intermediate analytical units (SOIs).

Examples of Lens include:

- a cluster from Diversity
- a Top-N set from Preference
- an efficient subset
- a non-dominated subset
- a manually saved current set
""",

    "saved_sois": """
Review, load, or delete saved Solutions of Interest.

Saved SOIs store:

- solution IDs
- source lens
- method
- selected group
- parameters
- creation context

Use saved SOIs to return to previous interesting subsets or combine them later.
""",

    "css": """
Define the Candidate Solution Set.

The CSS is the final subset that will be studied in detail.

You can build it from:

- the current decision set
- the current SOI
- a manual selection of specific solutions

Once a CSS is active, it can be used for detailed visual comparison.
""",

    "summary": """
Summarize the current decision set or CSS.

This section shows:

- number of solutions
- number of attributes
- number of decision variables
- CSS status
- derived columns
- export options
- the current data table
""",

    "maps": """
Explore the current decision set or CSS visually.

Maps help reveal:

- trade-offs
- clusters
- consensus groups
- preference scores
- efficiency scores
- highlighted candidates
""",

    "comparison": """
Compare selected candidate solutions in detail.

The detailed comparison includes:

- radar profiles for objectives and indicators
- decision-variable composition matrix
- decision-variable distribution inside the CSS
"""
}


# =====================================================
# HELP ACCESS
# =====================================================

def get_phase_help(
    phase_key
):

    return PHASE_HELP.get(
        phase_key,
        ""
    )


def get_help_text(
    phase_key
):

    return get_phase_help(
        phase_key
    )


# =====================================================
# GENERIC HELP POPOVER
# =====================================================

def render_help_icon(
    help_text,
    key=None,
    label="ⓘ"
):

    if not help_text:

        return

    with st.popover(
        label
    ):

        st.markdown(
            help_text
        )


# =====================================================
# PHASE HELP POPOVER
# =====================================================

def render_phase_help_icon(
    phase_key,
    key=None,
    label="ⓘ"
):

    help_text = get_phase_help(
        phase_key
    )

    if not help_text:

        return

    render_help_icon(
        help_text,
        key=key,
        label=label
    )