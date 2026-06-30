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
