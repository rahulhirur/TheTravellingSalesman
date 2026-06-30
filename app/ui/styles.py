import streamlit as st
import os

# Solvers styling configurations
COLORS = {
    "or_tools": "#3b82f6",
    "christofides": "#8b5cf6",
    "nearest_neighbor": "#f97316",
    "exact": "#10b981",
    "notebook_tf": "#ec4899",
    "notebook_tf_ga": "#f43f5e",
    "ground_truth": "#eab308"
}

SOLVER_NAMES = {
    "or_tools": "Google OR-Tools Heuristic",
    "christofides": "Christofides Approximation",
    "nearest_neighbor": "Nearest Neighbor Heuristic",
    "exact": "Held-Karp Exact Solver",
    "notebook_tf": "Notebook Transformer (Standard)",
    "notebook_tf_ga": "Notebook Transformer (Grad Accum)",
    "ground_truth": "Ground Truth (Optimal Tour)"
}

def inject_custom_css(is_dark: bool):
    bg_color = "#0b0d13" if is_dark else "#f9fafb"
    panel_bg = "rgba(22, 26, 37, 0.75)" if is_dark else "#ffffff"
    panel_border = "rgba(255, 255, 255, 0.08)" if is_dark else "#e4e4e7"
    text_color = "#f0f2f5" if is_dark else "#18181b"
    text_muted = "#9aa2b1" if is_dark else "#71717a"
    card_bg = "rgba(13, 16, 23, 0.9)" if is_dark else "#f4f4f5"
    glow_shadow = "rgba(6, 182, 212, 0.2)" if is_dark else "rgba(6, 182, 212, 0.05)"
    th_bg = "rgba(0, 0, 0, 0.2)" if is_dark else "rgba(0, 0, 0, 0.03)"
    td_border = "rgba(255, 255, 255, 0.03)" if is_dark else "rgba(0, 0, 0, 0.03)"
    tr_hover = "rgba(255, 255, 255, 0.02)" if is_dark else "rgba(0, 0, 0, 0.015)"

    css_styles = f"""
    <style>
        /* Hide Streamlit chrome */
        header[data-testid="stHeader"], footer, .stDeployButton {{
            display: none !important;
        }}
        
        /* Global styles */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
            background-color: {bg_color} !important;
            color: {text_color} !important;
            font-family: 'Outfit', sans-serif !important;
        }}
        
        .block-container {{
            padding: 1.5rem 2rem 2rem !important;
        }}
        
        /* Panel Cards */
        .glass-card, div[data-testid="stVerticalBlockBorder"] {{
            background: {panel_bg} !important;
            border: 1px solid {panel_border} !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            margin-bottom: 1.5rem !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
            backdrop-filter: blur(10px) !important;
        }}
        
        .glass-header {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: {text_color};
            border-bottom: 1px solid {panel_border};
            padding-bottom: 0.5rem;
        }}
        
        /* Metric KPI Card */
        .metric-card {{
            background: {card_bg};
            border: 1px solid {panel_border};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-label {{
            font-size: 0.78rem;
            color: {text_muted};
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .metric-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: {text_color};
            margin-top: 0.2rem;
        }}
        .metric-delta {{
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.4rem;
            padding: 2px 8px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 3px;
        }}
        .delta-best {{
            color: #10b981;
            background: rgba(16, 185, 129, 0.12);
        }}
        .delta-gap {{
            color: #ef4444;
            background: rgba(239, 68, 68, 0.12);
        }}
        
        /* HTML Data Table */
        .data-table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}
        .data-table th {{
            text-align: left;
            padding: 0.75rem 1rem;
            color: {text_muted};
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid {panel_border};
            background: {th_bg};
        }}
        .data-table td {{
            padding: 0.75rem 1rem;
            color: {text_color};
            border-bottom: 1px solid {td_border};
        }}
        .data-table tr:last-child td {{
            border-bottom: none;
        }}
        .data-table tr:hover td {{
            background: {tr_hover};
        }}
        .color-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        
        /* Interactive SVG diagram styles */
        .arch-svg {{
            width: 100%;
            height: auto;
        }}
        .arch-node {{
            cursor: pointer;
            transition: all 0.2s;
        }}
        .arch-node:hover {{
            filter: drop-shadow(0 0 6px {glow_shadow});
        }}
        .arch-edge {{
            stroke: rgba(255, 255, 255, 0.15);
            stroke-width: 1.5;
            fill: none;
            stroke-dasharray: 4 2;
        }}
        .dimension-badge {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            color: {text_color};
            background: rgba(255, 255, 255, 0.08);
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 500;
        }}
        
        .status-badge {{
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .status-online {{ background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.2); }}
        .status-offline {{ background: rgba(249, 115, 22, 0.12); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.2); }}
    </style>
    """
    st.markdown(css_styles, unsafe_allow_html=True)

def render_app_header(is_dark: bool, text_muted: str):
    title_color_start = "#1e293b" if not is_dark else "#ffffff"
    title_color_end = "#64748b" if not is_dark else "#9aa2b1"
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 1.5rem;">
        <span style="font-size: 32px; color: #06b6d4; text-shadow: 0 0 10px rgba(6,182,212,0.4)">☍</span>
        <div>
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, {title_color_start} 40%, {title_color_end} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TSP Solver MLOps Platform</h1>
            <p style="margin: 0; font-size: 0.85rem; color: {text_muted};">Transformer Sequence Models vs Conventional Graph Optimization Heuristics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
