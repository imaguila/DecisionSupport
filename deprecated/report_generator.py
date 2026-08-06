## --------------------------------------------------------------------------------------
## report_generator.py
## --------------------------------------------------------------------------------------

import pandas as pd
from datetime import datetime

def generate_markdown_report(dataset, css_df, saved_sois, highlight_ids):
    """
    Genera un informe ejecutivo consolidado en formato Markdown.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_name = dataset.get("config", {}).get("name", "Pareto Optimization Dataset")
    
    report = []
    
    # 1. Header
    report.append(f"# 📊 Executive Decision Report: {dataset_name}")
    report.append(f"**Generated on:** {timestamp}\n")
    report.append("---")
    
    # 2. Executive Overview
    report.append("## 1. Executive Overview")
    total_css = len(css_df) if css_df is not None else 0
    total_sois = len(saved_sois) if saved_sois else 0
    total_highlighted = len(highlight_ids) if highlight_ids else 0
    
    report.append(f"- **Candidate Solution Set (CSS) Size:** {total_css} solutions")
    report.append(f"- **Saved Sets of Interest (SOIs):** {total_sois} sets")
    report.append(f"- **Highlighted Solutions for Deep Dive:** {total_highlighted} solutions\n")
    
    # 3. Saved SOIs Summary
    report.append("## 2. Analyzed Sets of Interest (SOIs)")
    if saved_sois:
        report.append("| SOI Name | Lens Type | Size | Source Group |")
        report.append("| :--- | :--- | :--- | :--- |")
        for soi in saved_sois:
            name = soi.get("name", "Unnamed")
            lens = soi.get("lens", "Manual/Direct")
            size = soi.get("soi_size", len(soi.get("ids", [])))
            group = soi.get("group", "N/A")
            report.append(f"| {name} | {lens} | {size} | {group} |")
    else:
        report.append("_No SOIs were explicitly saved during this session._")
    report.append("\n")

    # 4. Highlighted Solutions Comparison
    report.append("## 3. Top Candidate Solutions Comparison")
    if css_df is not None and highlight_ids:
        highlighted_df = css_df[css_df["id"].isin(highlight_ids)].copy()
        metrics = dataset.get("metrics", []) + dataset.get("selected_indicators", [])
        available_metrics = [m for m in metrics if m in highlighted_df.columns]
        
        cols_to_show = ["id"] + available_metrics
        subset_df = highlighted_df[cols_to_show]
        
        report.append(subset_df.to_markdown(index=False))
    else:
        report.append("_No specific solutions were highlighted for direct comparison._")
    report.append("\n")

    # 5. Conclusion / Footer
    report.append("---")
    report.append("*Report generated automatically by Pareto Framework Decision Tool.*")
    
    return "\n".join(report)