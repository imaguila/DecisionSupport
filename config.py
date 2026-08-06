"""
Example configurations shipped with the framework.

A configuration describes:
1. How a Pareto front is loaded.
2. Which columns correspond to optimization objectives.
3. How decision variables are identified.
4. Which domain plugin should be used.
5. Which enrichment indicators are available by default.

Users may:
- use an existing configuration,
- define their own configuration,
- or provide an already enriched dataset with no plugin.
"""

from typing import Dict, List, Optional, TypedDict


class CaseConfig(TypedDict, total=False):
    """Schema definition for domain case configurations."""

    plugin: str
    path_sol: str
    metrics: List[str]
    var_prefix: str
    num_x: int
    exclude_cols: List[str]
    default_indicators: List[str]
    help: str


# =====================================================================
# CONFIGURATION REGISTRY
# =====================================================================

CASES: Dict[str, CaseConfig] = {
    # -----------------------------------------------------------------
    # CASE 1: Software Release Planning - CLASSIC Dataset
    # -----------------------------------------------------------------
    "CLASSIC Dataset": {
        "plugin": "nrp",
        "path_sol": "data/bagnallsoluciones.csv",
        "metrics": ["satisfaction", "effort"],
        "var_prefix": "req_",
        "num_x": 18,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": ["scope", "productivity", "squandering"],
        "help": (
            "Greer, D., & Ruhe, G. (2004). Software release planning: an evolutionary "
            "and iterative approach. Information and Software Technology, 46(4), 243-253."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 2: Software Release Planning - MSLite System
    # -----------------------------------------------------------------
    "MSLite System": {
        "plugin": "nrp",
        "path_sol": "data/mslitesoluciones.csv",
        "metrics": ["satisfaction", "effort", "dissatisfaction"],
        "var_prefix": "req_",
        "num_x": 16,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering",
            "annoyance",
            "dirtiness",
        ],
        "help": (
            "Sangwan, R. S., Negahban, A., Nord, R. L., & Ozkaya, I. (2020). "
            "Optimization of software release planning considering architectural "
            "dependencies, cost, and value. IEEE Transactions on Software Engineering, "
            "48(4), 1369-1384."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 3: Replacement Access, Library and ID Card (RALIC)
    # -----------------------------------------------------------------
    "Replacement Access, Library and ID Card (RALIC)": {
        "plugin": "nrp",
        "path_sol": "data/ralic.csv",
        "metrics": ["satisfaction", "effort"],
        "var_prefix": "req_",
        "num_x": 83,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": ["scope", "productivity", "squandering"],
        "help": (
            "Lim, S. L., & Finkelstein, A. (2011). StakeRare: using social networks "
            "and collaborative filtering for large-scale requirements elicitation. "
            "IEEE Transactions on Software Engineering, 38(3), 707-735."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 4: Word Processing Software Project
    # -----------------------------------------------------------------
    "Word Processing Software Project": {
        "plugin": "nrp",
        "path_sol": "data/wordprocsoluciones.csv",
        "metrics": ["satisfaction", "effort", "time"],
        "var_prefix": "req_",
        "num_x": 42,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering",
            "response",
            "opportunity",
        ],
        "help": (
            "Agarwal, N., Karimpour, R., & Ruhe, G. (2014). Theme-based product "
            "release planning: An analytical approach. In 2014 47th Hawaii International "
            "Conference on System Sciences, pp. 4739-4748. IEEE."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 5: Large Dataset - REQ100
    # -----------------------------------------------------------------
    "Large Dataset": {
        "plugin": "nrp",
        "path_sol": "data/req100frente.csv",
        "metrics": ["satisfaction", "effort"],
        "var_prefix": "req_",
        "num_x": 96,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": ["scope", "productivity", "squandering"],
        "help": (
            "Del Sagrado, J., Del Águila, I. M., & Orellana, F. J. (2015). Multi-objective "
            "ant colony optimization for requirements selection. Empirical Software "
            "Engineering, 20(3), 577-610."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 6: ReleasePlanner Dataset - THEME
    # -----------------------------------------------------------------
    "ReleasePlanner™ Dataset": {
        "plugin": "nrp",
        "path_sol": "data/themesoluciones.csv",
        "metrics": [
            "satisfaction",
            "prevalence",
            "cost",
            "dissatisfaction",
            "inestability",
            "effort",
        ],
        "var_prefix": "req_",
        "num_x": 22,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": [
            "scope",
            "productivity",
            "squandering",
            "effectiveness",
            "dirtiness",
            "annoyance",
            "stickiness",
            "fragility",
            "robustness",
            "usage_efficiency",
        ],
        "help": (
            "Karim, M. R., & Ruhe, G. (2014). Bi-objective genetic search for release "
            "planning in support of themes. In International Symposium on Search Based "
            "Software Engineering, pp. 123-137. Springer International Publishing."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 7: Motorola Dataset
    # -----------------------------------------------------------------
    "Motorola Dataset": {
        "plugin": "nrp",
        "path_sol": "data/motorolasoluciones.csv",
        "metrics": ["satisfaction", "effort"],
        "var_prefix": "req_",
        "num_x": 35,
        "exclude_cols": ["id", "run_id", "timestamp"],
        "default_indicators": ["scope", "productivity", "squandering"],
        "help": (
            "Baker, P., Harman, M., Steinhofel, K., & Skaliotis, A. (2006). Search based "
            "approaches to component selection and prioritization for the next release "
            "problem. In 2006 22nd IEEE International Conference on Software Maintenance, "
            "pp. 176-185. IEEE."
        ),
    },
    # -----------------------------------------------------------------
    # CASE 8: Generic Engineering Design - Aerospace Wing Design
    # -----------------------------------------------------------------
    "Aerospace Wing Design": {
        "plugin": "aerospace",
        "path_sol": "data/wing_pareto_front.csv",
        "metrics": ["drag", "weight"],
        "var_prefix": "var_",
        "num_x": 10,
        "exclude_cols": ["sim_time", "solver_status"],
        "default_indicators": [
            "density",
            "lift_to_drag_ratio",
            "structural_efficiency",
        ],
        "help": (
            "Example, A. et al. (2025). Multi-objective aerodynamic design optimization "
            "of aircraft wings. Journal of Aircraft, 62(1), 100-115."
        ),
    },
}


# =====================================================================
# ACCESSOR UTILITIES
# =====================================================================


def get_available_cases() -> List[str]:
    """
    Returns a list of all pre-configured domain case names.

    Returns
    -------
    List[str]
        Names of all available cases in the registry.
    """
    return list(CASES.keys())


def get_case_config(case_name: str) -> Optional[CaseConfig]:
    """
    Safely retrieves the configuration dictionary for a target case name.

    Parameters
    ----------
    case_name : str
        Name of the case entry.

    Returns
    -------
    Optional[CaseConfig]
        Configuration metadata dictionary, or None if case name does not exist.
    """
    return CASES.get(case_name)