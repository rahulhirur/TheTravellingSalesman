import streamlit as st
import torch
import torch.nn as nn
from torchview import draw_graph

st.title("Transformer Encoder-Decoder Explorer")

# --- Sidebar Parameters ---
st.sidebar.subheader("Model Parameters")
d_model = st.sidebar.slider("Model Dimension (d_model)", 64, 512, 128, step=64)
nhead = st.sidebar.selectbox("Attention Heads", [2, 4, 8], index=1)
num_encoder_layers = st.sidebar.slider("Encoder Layers", 1, 3, 1)
num_decoder_layers = st.sidebar.slider("Decoder Layers", 1, 3, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Visualization Settings")
visualization_depth = st.sidebar.slider("Graph Depth", 1, 5, 2)
expand_nested = st.sidebar.checkbox("Expand Nested Modules", value=True)

# --- 1. S2S Transformer Definition (Standard / Pre-built Layers) ---
class DummyTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_enc_layers, num_dec_layers):
        super().__init__()
        # Standard PyTorch layers (No custom subclassing here)
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

# Instantiate standard model
model = DummyTransformer(
    d_model=d_model,
    nhead=nhead,
    num_enc_layers=num_encoder_layers,
    num_dec_layers=num_decoder_layers
)

# --- 2. Dynamic Renaming Helper for Pre-built / Standard Models ---
# Recursively traverses and renames submodule class types dynamically at runtime.
def rename_model_layers(module_to_rename):
    for name, submodule in module_to_rename.named_modules():
        if name == "":
            continue # Skip root module itself
            
        # Create a clean label class name from the module path
        # e.g., "encoder.layers.0" -> "encoder_layers_0"
        clean_name = name.replace('.', '_')
        
        try:
            # Dynamically subclass the submodule's class at runtime
            custom_class = type(clean_name, (submodule.__class__,), {})
            submodule.__class__ = custom_class
        except Exception:
            # Catch cases where __class__ reassignment is restricted (e.g. C-level classes)
            pass

# Apply the dynamic renamer before feeding to torchview!
rename_model_layers(model)

# --- 3. Generate Computation Graph ---
dummy_src = torch.randn(1, 10, 2)
dummy_tgt = torch.randn(1, 10, 2)

with st.spinner("Generating computation graph..."):
    try:
        model_graph = draw_graph(
            model,
            input_data=(dummy_src, dummy_tgt),
            expand_nested=expand_nested,
            depth=visualization_depth
        )
        
        st.graphviz_chart(model_graph.visual_graph.source)
        st.success("Successfully visualized dynamically-named pre-built model graph!")
    except Exception as e:
        st.error(f"Failed generating graph: {e}")
