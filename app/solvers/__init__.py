try:
    from app.solvers.conventional import (
        solve_nearest_neighbor,
        solve_christofides,
        solve_or_tools,
        solve_exact,
        calculate_path_distance
    )
except ImportError:
    solve_nearest_neighbor = None
    solve_christofides = None
    solve_or_tools = None
    solve_exact = None
    calculate_path_distance = None

from app.solvers.validation import validate_tsp_coordinates
from app.solvers.dataset import (
    load_pkl_dataset,
    convert_to_networkx_graph,
    save_pkl_dataset
)

try:
    from app.solvers.notebook_transformer import solve_notebook_transformer
except ImportError:
    solve_notebook_transformer = None

__all__ = [
    "solve_nearest_neighbor",
    "solve_christofides",
    "solve_or_tools",
    "solve_exact",
    "solve_notebook_transformer",
    "calculate_path_distance",
    "validate_tsp_coordinates",
    "load_pkl_dataset",
    "convert_to_networkx_graph",
    "save_pkl_dataset"
]
