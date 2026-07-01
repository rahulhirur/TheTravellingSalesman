import os
import time
import math
import torch
import torch.nn as nn
import numpy as np

# Corrected PositionalEncoding class from notebook
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0) # Change to (1, maxlen, emb_size) for broadcasting

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1), :])

# Custom transformer model for TSP from notebook
class TSP_Transformer(nn.Module):
    def __init__(self, num_cities, d_model_enc, d_model_dec, d_model_ff, nhead, num_layers_enc, num_layers_dec, dropout_rate=0.3):
        super(TSP_Transformer, self).__init__()
        self.num_cities = num_cities

        self.cartesian_embedding = nn.Linear(2, d_model_enc)
        self.city_embedding = nn.Embedding(num_cities, d_model_dec)
        self.positional_encoding_dec = PositionalEncoding(d_model_dec, dropout=dropout_rate, maxlen=num_cities)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model_enc, nhead, dim_feedforward=d_model_ff, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers_enc)
        self.encode_to_decoder_prj = nn.Linear(d_model_enc, d_model_dec)

        decoder_layers = nn.TransformerDecoderLayer(
            d_model_dec, nhead, dim_feedforward=d_model_ff, dropout=dropout_rate, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers_dec)
        self.output_layer = nn.Linear(d_model_dec, num_cities)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, coords, initial_order):
        cartesian_embeddings = self.cartesian_embedding(coords)
        encoder_output = self.transformer_encoder(cartesian_embeddings)
        encoder_output = self.encode_to_decoder_prj(encoder_output)
        city_embeddings = self.city_embedding(initial_order)
        city_embeddings = self.positional_encoding_dec(city_embeddings)
        
        seq_len = initial_order.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(coords.device)
        decoder_output = self.transformer_decoder(city_embeddings, encoder_output, tgt_mask=mask)
        output = self.output_layer(decoder_output)
        return output

# Cached models
_cached_models = {}

def get_notebook_model(use_ga=False):
    global _cached_models
    key = "ga" if use_ga else "standard"
    if key in _cached_models:
        return _cached_models[key]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TSP_Transformer(
        num_cities=20, 
        d_model_enc=32, 
        d_model_dec=32, 
        d_model_ff=64, 
        nhead=8, 
        num_layers_enc=3, 
        num_layers_dec=3, 
        dropout_rate=0.3
    )
    
    # Path to saved weights relative to project root
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_filename = "model_transformer_GA.pth" if use_ga else "model_transformer.pth"
    weights_path = os.path.join(parent_dir, "misc", weights_filename)
    
    # Try local load first. If missing, attempt downloading from HF Hub (public or private repository)
    if not os.path.exists(weights_path):
        hf_repo = os.getenv("HF_MODEL_REPO")
        hf_token = os.getenv("HF_TOKEN")
        if hf_repo:
            try:
                from huggingface_hub import hf_hub_download
                print(f"Downloading {weights_filename} from HF Repo {hf_repo}...")
                cache_dir = os.path.join(parent_dir, "misc", ".cache")
                downloaded_path = hf_hub_download(
                    repo_id=hf_repo,
                    filename=weights_filename,
                    token=hf_token,
                    local_dir=os.path.join(parent_dir, "misc"),
                    cache_dir=cache_dir
                )
                if downloaded_path and os.path.exists(downloaded_path):
                    weights_path = downloaded_path
            except Exception as e:
                print(f"Failed downloading weights from Hugging Face Hub: {e}")
    
    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded notebook model {key} weights from {weights_path}")
        except Exception as e:
            print(f"Failed loading {weights_path}: {e}")
    else:
        print(f"Warning: weights file {weights_filename} not found locally or on HF Hub. Running on random weights.")
    
    model = model.to(device)
    _cached_models[key] = model
    return model

def solve_notebook_transformer(coords, use_ga=False):
    """Inference for notebook TSP Transformer with outer search loop filtering."""
    start_time = time.perf_counter()
    n = len(coords)
    
    # If N is not 20, fallback to nearest neighbor to avoid IndexErrors in embedding
    if n != 20:
        from app.solvers.conventional import solve_nearest_neighbor
        return solve_nearest_neighbor(coords)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_notebook_model(use_ga)
    model.eval()
    
    # Input coords normalized
    x = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0) # (1, N, 2)
    
    tour = [0]
    y = torch.tensor(tour, dtype=torch.long, device=device).unsqueeze(0) # (1, 1)
    
    with torch.no_grad():
        out = model(x, y)
        
        while len(tour) < n:
            _, idx = torch.topk(out, n, dim=2)
            # Find the highest probability city that is not yet visited
            for i in range(n):
                city_cand = int(idx[0, -1, i].item())
                if city_cand not in tour:
                    tour.append(city_cand)
                    break
            else:
                # Fallback in case loop fails
                for c in range(n):
                    if c not in tour:
                        tour.append(c)
                        break
                        
            y = torch.tensor(tour, dtype=torch.long, device=device).unsqueeze(0)
            out = model(x, y)
            
    tour.append(0) # Complete loop back to start
    
    # Compute path distance
    dist = 0.0
    for i in range(n):
        c1 = coords[tour[i]]
        c2 = coords[tour[i+1]]
        dist += math.hypot(c1[0] - c2[0], c1[1] - c2[1])
        
    elapsed = time.perf_counter() - start_time
    return tour, dist, elapsed
