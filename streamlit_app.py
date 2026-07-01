import os
import time
import math
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st

# Import UI helpers
from app.ui.styles import COLORS, SOLVER_NAMES, inject_custom_css, render_app_header
from app.ui.components import render_plotly_route, render_graphviz_trace
from app.solvers.dataset import load_pkl_dataset

# 1. Page Configuration
st.set_page_config(
    page_title="TSP Solver: Neural vs Heuristics",
    page_icon="☍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Session State Initialization
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "cities" not in st.session_state:
    # Default initial set of 20 random cities
    np.random.seed(42)
    st.session_state.cities = np.random.uniform(50, 950, size=(20, 2)).tolist()

if "results" not in st.session_state:
    st.session_state.results = {}

# 3. App Header
inject_custom_css(False)
text_muted = "#71717a"
render_app_header(False, text_muted)

# 4. API Connection Status Helper
api_base_url = os.getenv("BACKEND_API_URL")
if not api_base_url:
    space_id = os.getenv("SPACE_ID")
    if space_id and "/" in space_id:
        username = space_id.split("/")[0].lower().replace("_", "-")
        api_base_url = f"https://{username}-thetravellingsalesman-api.hf.space"
    else:
        api_base_url = "http://localhost:7860"

hf_token = os.getenv("HF_TOKEN")

@st.cache_data(ttl=2)
def check_api_health(url):
    try:
        headers = {}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        res = requests.get(f"{url}/health", headers=headers, timeout=2.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, None

api_online, health_data = check_api_health(api_base_url)

# Setup navigation tabs
tab1, tab3 = st.tabs(["🎮 Solver Playground", "🧠 Architecture Hub"])

# ==========================================
# 1. PLAYGROUND TAB
# ==========================================
with tab1:
    col_ctrl, col_viz = st.columns([4, 8])
    
    with col_ctrl:
        with st.container(border=True):
            st.markdown('<div class="glass-header">Solver Controls</div>', unsafe_allow_html=True)
            
            # API Connection
            st.markdown("**Backend Service Status**")
            if api_online:
                st.markdown('<span class="status-badge status-online">Online</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-offline">Offline: Waking Server...</span>', unsafe_allow_html=True)
            
            # Graph Generator & Data Source Selection
            st.write("---")
            st.markdown("**Data Source Selection**")
            data_source = st.selectbox(
                "Input Data Source",
                ["Random City Generator", "Hugging Face Dataset"],
                key="data_source_select"
            )
            
            if data_source == "Random City Generator":
                num_nodes = st.slider("Number of Cities (N)", min_value=5, max_value=100, value=len(st.session_state.cities), step=1)
                
                col_gen1, col_gen2 = st.columns(2)
                with col_gen1:
                    if st.button("Generate Random", use_container_width=True, key="btn_gen_random"):
                        np.random.seed(int(time.time()))
                        st.session_state.cities = np.random.uniform(50, 950, size=(num_nodes, 2)).tolist()
                        st.session_state.results = {}
                        st.rerun()
                with col_gen2:
                    if st.button("Reset Seed (42)", use_container_width=True, key="btn_reset_seed"):
                        np.random.seed(42)
                        st.session_state.cities = np.random.uniform(50, 950, size=(num_nodes, 2)).tolist()
                        st.session_state.results = {}
                        st.rerun()
            else:
                headers = {}
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                
                try:
                    res = requests.get(f"{api_base_url}/datasets", headers=headers, timeout=5.0)
                    if res.status_code == 200:
                        pkl_files = res.json().get("datasets", [])
                    else:
                        pkl_files = []
                except Exception:
                    pkl_files = []
                
                if pkl_files:
                    selected_file = st.selectbox("Select .pkl dataset file", pkl_files)
                    sample_idx = st.number_input("Sample Index", min_value=0, max_value=999, value=0, step=1)
                    
                    if st.button("Load Dataset Sample", use_container_width=True, key="btn_load_sample") or "last_loaded_sample" not in st.session_state or st.session_state.last_loaded_sample != (selected_file, sample_idx):
                        try:
                            sample_res = requests.get(f"{api_base_url}/dataset/{selected_file}/{sample_idx}", headers=headers, timeout=5.0)
                            if sample_res.status_code == 200:
                                sample_data = sample_res.json()
                                coords_sample = sample_data["points"]
                                gt_path = sample_data["ground_truth"]
                                
                                scaled_coords = np.array(coords_sample) * 1000.0
                                st.session_state.cities = scaled_coords.tolist()
                                
                                if len(gt_path) > 0 and gt_path[-1] != gt_path[0]:
                                    gt_path.append(gt_path[0])
                                    
                                dist = 0.0
                                for i in range(len(gt_path) - 1):
                                    c1 = scaled_coords[gt_path[i]]
                                    c2 = scaled_coords[gt_path[i+1]]
                                    dist += math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                                    
                                st.session_state.results["ground_truth"] = {
                                    "path": gt_path,
                                    "distance": dist,
                                    "time_taken": 0.0
                                }
                                st.session_state.last_loaded_sample = (selected_file, sample_idx)
                                st.success("Loaded dataset sample successfully!")
                                st.rerun()
                            else:
                                st.error(f"Failed to load dataset sample: {sample_res.text}")
                        except Exception as e:
                            st.error(f"Error connecting to backend: {e}")
                else:
                    st.warning("No valid .pkl files found in Hugging Face Model Repository. Upload .pkl files to your model repo to see them here!")
                
                num_nodes = len(st.session_state.cities)
            
            # Solver selections
            st.write("---")
            st.markdown("**Select Algorithms**")
            sel_ortools = st.checkbox("Google OR-Tools", value=True)
            sel_christofides = st.checkbox("Christofides Approximation", value=True)
            sel_greedy = st.checkbox("Nearest Neighbor Heuristic", value=True)
            sel_exact = st.checkbox("Held-Karp Exact Solver", value=False, disabled=(num_nodes > 10), 
                                    help="True optimum. Hard limit N <= 10 to avoid performance freezes.")
            sel_notebook_tf = st.checkbox("Notebook Transformer (Standard)", value=(num_nodes == 20), disabled=(num_nodes != 20),
                                          help="Only supports N = 20 nodes as designed in assignment.")
            sel_notebook_tf_ga = st.checkbox("Notebook Transformer (Grad Accum)", value=False, disabled=(num_nodes != 20),
                                             help="Only supports N = 20 nodes as designed in assignment.")
            
            # Execution action
            if not api_online:
                st.warning("⚠️ Backend API is offline or sleeping. Wake-up request sent. Please wait about 30 seconds for it to boot.")
                if st.button("🔄 Check Connection Status", use_container_width=True, key="btn_retry_conn"):
                    st.rerun()
            
            btn_disabled = not api_online
            btn_label = "🚀 Solve Travelling Salesman" if api_online else "⏳ Waking Backend API..."
            if st.button(btn_label, use_container_width=True, type="primary", key="btn_solve_tsp", disabled=btn_disabled):
                gt_data = st.session_state.results.get("ground_truth")
                st.session_state.results = {}
                if gt_data:
                    st.session_state.results["ground_truth"] = gt_data
                    
                selected_solvers = []
                if sel_ortools: selected_solvers.append("or_tools")
                if sel_christofides: selected_solvers.append("christofides")
                if sel_greedy: selected_solvers.append("nearest_neighbor")
                if sel_exact and num_nodes <= 10: selected_solvers.append("exact")
                if sel_notebook_tf and num_nodes == 20: selected_solvers.append("notebook_tf")
                if sel_notebook_tf_ga and num_nodes == 20: selected_solvers.append("notebook_tf_ga")
                
                if not selected_solvers:
                    st.warning("Please check at least one solver.")
                else:
                    gt_data = st.session_state.results.get("ground_truth")
                    st.session_state.results = {}
                    if gt_data:
                        st.session_state.results["ground_truth"] = gt_data
                    
                    coords_scaled = np.array(st.session_state.cities)
                    
                    # Request through Backend API only
                    try:
                        points_norm = (coords_scaled / 1000.0).tolist()
                        headers = {}
                        if hf_token:
                            headers["Authorization"] = f"Bearer {hf_token}"
                        res = requests.post(f"{api_base_url}/solve", json={
                            "points": points_norm,
                            "solvers": selected_solvers
                        }, headers=headers)
                        if res.status_code == 200:
                            api_results = res.json()["results"]
                            for k, v in api_results.items():
                                st.session_state.results[k] = {
                                    "path": v["path"],
                                    "distance": v["distance"] * 1000.0,
                                    "time_taken": v["time_taken"]
                                }
                        else:
                            st.error(f"API Solver error: {res.text}")
                    except Exception as err:
                        st.error(f"API Connection error during solving: {err}")
    
    with col_viz:
        with st.container(border=True):
            st.markdown('<div class="glass-header">Interactive Route Visualizer</div>', unsafe_allow_html=True)
            
            # Display route graph using Plotly helper
            fig = render_plotly_route(st.session_state.cities, st.session_state.results, COLORS, SOLVER_NAMES)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            
        with st.container(border=True):
            st.markdown('<div class="glass-header">Solver Performance Diagnostics</div>', unsafe_allow_html=True)
            
            if st.session_state.results:
                # Find metrics
                valid_dists = {k: v["distance"] for k, v in st.session_state.results.items() if k != "ground_truth"}
                min_dist = min(valid_dists.values()) if valid_dists else 0.0
                best_solver = min(valid_dists, key=valid_dists.get) if valid_dists else None
                
                # Find best parameters, excluding ground truth from speed benchmarking
                active_solvers = [k for k in st.session_state.results.keys() if k != "ground_truth"]
                if active_solvers:
                    best_time_solver = min(active_solvers, key=lambda k: st.session_state.results[k]["time_taken"])
                else:
                    best_time_solver = None
                
                table_rows = ""
                for key, res in st.session_state.results.items():
                    is_best_dist = (key == best_solver)
                    gap = 0.0
                    if not is_best_dist and min_dist > 0:
                        gap = ((res["distance"] - min_dist) / min_dist) * 100
                    gap_text = "0.00%" if is_best_dist or key == "ground_truth" else f"+{gap:.2f}%"
                    
                    if key == "ground_truth":
                        badge_class = "delta-best"
                        badge_text = "Optimal"
                        latency_text = "N/A (Loaded)"
                    else:
                        badge_class = "delta-best" if is_best_dist else "delta-gap"
                        badge_text = "Shortest" if is_best_dist else "Suboptimal"
                        latency_text = f"{res['time_taken']*1000:.2f} ms"
                    
                    table_rows += f'<tr><td><span class="color-indicator" style="background-color: {COLORS.get(key, "#ffffff")}"></span><strong>{SOLVER_NAMES.get(key, key)}</strong></td><td>{res["distance"]:.4f}</td><td>{latency_text}</td><td>{gap_text}</td><td><span class="metric-delta {badge_class}">{badge_text}</span></td></tr>'
            
                table_html = f'<table class="data-table"><thead><tr><th>Solver</th><th>Path Distance</th><th>Latency (ms)</th><th>Accuracy Gap</th><th>Status</th></tr></thead><tbody>{table_rows}</tbody></table>'
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("Solve the TSP to see benchmarking diagnostics and comparative paths.")

# ==========================================
# 3. ARCHITECTURE HUB TAB
# ==========================================
with tab3:
    st.subheader("Interactive Layer Explorer")
    
    # 1. Config expander for live parameters
    with st.expander("⚙️ Model Architecture & Graph Tracer Settings", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            d_model_val = st.slider("Model Dimension (d_model)", 64, 512, 128, step=64, key="arch_d_model")
            nhead_val = st.selectbox("Attention Heads", [2, 4, 8], index=1, key="arch_nhead")
            
        with col_c2:
            num_enc_layers = st.slider("Encoder Layers", 1, 3, 2, key="arch_enc_layers")
            num_dec_layers = st.slider("Decoder Layers", 1, 3, 2, key="arch_dec_layers")
            
        with col_c3:
            visualization_depth = st.slider("Trace Depth", 1, 4, 2, key="arch_depth")
            expand_nested = st.checkbox("Expand Nested Modules", value=True, key="arch_expand")
            
    st.write("---")
    st.subheader("Model Execution Graph Visualizer")
    
    # Instantiate and trace model via UI helper
    with st.spinner("Tracing PyTorch execution graph..."):
        try:
            dot_source = render_graphviz_trace(
                d_model_val, nhead_val, num_enc_layers, num_dec_layers,
                expand_nested, visualization_depth
            )
            st.graphviz_chart(dot_source)
            st.caption("💡 Use the mouse wheel to zoom and drag to pan the SVG graph natively.")
        except Exception as e:
            st.error(f"Failed tracing model graph: {e}")
            
    # 3. Technical specification text block
    st.write("---")
    st.subheader("Architectural Design Details")
    
    st.markdown(f"""
    <div class="glass-card" style="padding: 20px;">
        <h4 style="color: #8b5cf6; margin-bottom: 10px;">Seq2Seq Transformer (Notebook Architecture)</h4>
        <p style="font-size: 13px; color: {text_muted}; line-height: 1.6;">
            The <strong>Seq2Seq Transformer</strong> from the notebook uses a standard sequence translation formulation.
        </p>
        <ul style="font-size: 12px; color: {text_muted}; padding-left: 20px; line-height: 1.6;">
            <li><strong>Input projections</strong>: Independently embeds coordinates via <code>src_embed</code> and partial tour sequence indices via <code>tgt_embed</code>.</li>
            <li><strong>Sinusoidal PE</strong>: Adds positional encodings to decoder inputs, assuming sequence positions carry meaningful semantic ordering.</li>
            <li><strong>Transformer Decoder</strong>: Decodes the target sequence using causal mask (hiding future steps) and cross-attends with encoder representations.</li>
            <li><strong>Classification Flaw</strong>: Projects decoder outputs to logits across all possible cities. It lacks visited masking, allowing it to predict already visited cities (invalid TSP tours). Enforced via Python outer loop during search.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
