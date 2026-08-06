#  Decision Space Explorer





[Streamlit App](awerqwer)

decisionsupportlibrary/
├── core/
│   ├── analytics/            # 🧠 Lógica matemática/estadística pura (Pandas/Numpy)
│   │   ├── correlation.py
│   │   ├── similarity.py
│   │   └── baseline.py
│   ├── domain/               # 📦 Modelos de dominio y estado
│   │   ├── models.py         # ParetoDataset, Solution, SOI dataclasses
│   │   ├── state.py          # Centralización de st.session_state
│   │   └── soi_registry.py   # Registro y persistencia de SOIs (MOVIDO AQUÍ)
│   ├── exports/              # 📄 Generadores de informes/exportaciones
│   │   └── report_generator.py
│   └── workspace.py          # Coordinador principal de vistas
├── lenses/                   # 🔍 Lentes analíticas (Plugins)
│   ├── base_lens.py          # Clase base / Protocol para crear lentes
│   ├── lens_registry.py
│   ├── lens_preference.py
│   └── lens_diversity.py
├── ui/                       # 🎨 Componentes visuales reutilizables
│   ├── css_comparison.py
│   ├── workspace_summary.py
│   └── phase_help.py
└── streamlit_app.py




##  Overview

## 💻 Running Locally

To run this dashboard on your local machine:

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
