import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

try:
    import torch
    import torch.nn as nn
    from torchview import draw_graph
except ImportError:
    torch = None
    nn = None
    draw_graph = None

# Model architecture from the notebook (Seq2Seq Transformer)
if nn is not None:
    class NotebookSeq2SeqTransformer(nn.Module):
        def __init__(self, d_model=128, nhead=4, num_enc_layers=3, num_dec_layers=3):
            super().__init__()
            self.src_embed = nn.Linear(2, d_model)
            self.tgt_embed = nn.Linear(2, d_model)
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_enc_layers)
            
            decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_dec_layers)
            
            self.out_projection = nn.Linear(d_model, 2)

        def forward(self, src, tgt):
            src_emb = self.src_embed(src)
            tgt_emb = self.tgt_embed(tgt)
            memory = self.encoder(src_emb)
            out = self.decoder(tgt_emb, memory)
            return self.out_projection(out)

    def rename_model_layers(module_to_rename):
        for name, submodule in module_to_rename.named_modules():
            if name == "":
                continue
            clean_name = name.replace('.', '_')
            try:
                custom_class = type(clean_name, (submodule.__class__,), {})
                submodule.__class__ = custom_class
            except Exception:
                pass
else:
    NotebookSeq2SeqTransformer = None
    rename_model_layers = None

def render_plotly_route(cities, results, colors, names):
    fig = go.Figure()
    coords_arr = np.array(cities)
    
    # Draw Nodes
    fig.add_trace(go.Scatter(
        x=coords_arr[:, 0],
        y=coords_arr[:, 1],
        mode='markers+text',
        marker=dict(size=12, color='#06b6d4', line=dict(width=2, color='#ffffff')),
        text=[str(i) for i in range(len(cities))],
        textposition="top center",
        name="Cities",
        hoverinfo='text'
    ))
    
    # Draw Routes
    for solver_key, res in results.items():
        if solver_key == "ground_truth":
            line_style = dict(color=colors.get(solver_key), width=3, dash='dash')
        else:
            line_style = dict(color=colors.get(solver_key), width=2)
            
        path = res["path"]
        path_coords = coords_arr[path]
        
        fig.add_trace(go.Scatter(
            x=path_coords[:, 0],
            y=path_coords[:, 1],
            mode='lines',
            line=line_style,
            name=names.get(solver_key, solver_key),
            opacity=0.85,
            hoverinfo='name'
        ))
        
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=480,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def render_graphviz_trace(d_model_val, nhead_val, num_enc_layers, num_dec_layers, expand_nested, visualization_depth):
    if torch is None or draw_graph is None or NotebookSeq2SeqTransformer is None:
        raise ImportError("PyTorch or Torchview is not installed in the environment.")
        
    dummy_src = torch.randn(1, 10, 2)
    dummy_tgt = torch.randn(1, 10, 2)
    
    model_to_trace = NotebookSeq2SeqTransformer(
        d_model=d_model_val, 
        nhead=nhead_val, 
        num_enc_layers=num_enc_layers, 
        num_dec_layers=num_dec_layers
    )
    rename_model_layers(model_to_trace)
    
    model_graph = draw_graph(
        model_to_trace,
        input_data=(dummy_src, dummy_tgt),
        expand_nested=expand_nested,
        depth=visualization_depth
    )
    return model_graph.visual_graph.source
