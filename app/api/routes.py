from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import datetime
import os

from app.solvers import (
    solve_nearest_neighbor,
    solve_christofides,
    solve_or_tools,
    solve_exact,
    solve_notebook_transformer,
    validate_tsp_coordinates
)

router = APIRouter()

# ----------------- REQUEST/RESPONSE SCHEMAS -----------------
class SolveRequest(BaseModel):
    points: List[List[float]]
    solvers: List[str]

class SolverData(BaseModel):
    path: List[int]
    distance: float
    time_taken: float

class SolveResponse(BaseModel):
    results: Dict[str, SolverData]

@router.get("/")
def read_root():
    """Root endpoint for status check compliance."""
    return {"message": "TSP Solver API is running", "status": "healthy"}

def get_backend_datasets():
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    misc_dir = os.path.join(parent_dir, "misc")
    os.makedirs(misc_dir, exist_ok=True)
    
    # Recursively find local .pkl files inside misc/
    local_files = []
    for root, dirs, filenames in os.walk(misc_dir):
        for filename in filenames:
            if filename.endswith(".pkl"):
                rel_path = os.path.relpath(os.path.join(root, filename), misc_dir)
                local_files.append(rel_path.replace("\\", "/"))
    
    hf_repo = os.getenv("HF_MODEL_REPO")
    hf_token = os.getenv("HF_TOKEN")
    if hf_repo:
        try:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi(token=hf_token)
            files = api.list_repo_files(repo_id=hf_repo)
            pkl_files = [f for f in files if f.endswith(".pkl")]
            
            for f in pkl_files:
                dest_path = os.path.join(misc_dir, f)
                if not os.path.exists(dest_path):
                    print(f"Downloading dataset {f} from HF Repository...")
                    hf_hub_download(
                        repo_id=hf_repo,
                        filename=f,
                        token=hf_token,
                        local_dir=misc_dir
                    )
                    if f not in local_files:
                        local_files.append(f)
        except Exception as e:
            print(f"Failed listing/downloading datasets from Hugging Face Hub: {e}")
            
    return sorted(list(set(local_files)))

@router.get("/datasets")
def list_datasets():
    """Lists all available dataset files present in local or remote HF Model Repository."""
    try:
        files = get_backend_datasets()
        return {"datasets": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/{filename:path}/{sample_index}")
def get_dataset_sample(filename: str, sample_index: int):
    """Loads the specified dataset sample and returns normalized coordinates and ground truth."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(parent_dir, "misc", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Dataset {filename} not found.")
        
    try:
        from app.solvers.dataset import load_pkl_dataset
        coords_list, tours_list = load_pkl_dataset(file_path)
        
        if sample_index < 0 or sample_index >= len(coords_list):
            raise HTTPException(status_code=400, detail=f"Sample index {sample_index} out of range (0-{len(coords_list)-1}).")
            
        return {
            "points": coords_list[sample_index],
            "ground_truth": tours_list[sample_index]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

@router.get("/health")
def health():
    """Health check endpoint containing model loading states."""
    # Check if standard notebook transformer weights are cached or downloaded
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_loaded = os.path.exists(os.path.join(parent_dir, "misc", "model_transformer.pth"))
    
    return {
        "status": "healthy",
        "notebook_model_loaded": model_loaded,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@router.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    """Solves TSP on the coordinates using specified solvers, and benchmarks them."""
    try:
        validate_tsp_coordinates(request.points)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    n = len(request.points)
    
    if "exact" in request.solvers and n > 10:
        raise HTTPException(
            status_code=400, 
            detail="Held-Karp Exact Solver only supports N <= 10 nodes to avoid CPU timeouts."
        )
        
    results = {}
    
    for solver_name in request.solvers:
        try:
            if solver_name == "nearest_neighbor":
                path, dist, elapsed = solve_nearest_neighbor(request.points)
            elif solver_name == "or_tools":
                path, dist, elapsed = solve_or_tools(request.points)
            elif solver_name == "christofides":
                path, dist, elapsed = solve_christofides(request.points)
            elif solver_name == "exact":
                path, dist, elapsed = solve_exact(request.points)
            elif solver_name == "notebook_tf":
                path, dist, elapsed = solve_notebook_transformer(request.points, use_ga=False)
            elif solver_name == "notebook_tf_ga":
                path, dist, elapsed = solve_notebook_transformer(request.points, use_ga=True)
            else:
                continue
                
            results[solver_name] = SolverData(
                path=path,
                distance=dist,
                time_taken=elapsed
            )
        except Exception as e:
            # Prevent failure of one solver from crashing the entire batch response
            print(f"Error executing solver {solver_name}: {e}")
            continue
            
    return SolveResponse(results=results)
