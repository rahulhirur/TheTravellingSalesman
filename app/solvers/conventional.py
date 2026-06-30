import time
import math
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def calculate_path_distance(path, coords):
    """Calculates the total Euclidean distance of a given tour path."""
    if not path or len(path) <= 1:
        return 0.0
    dist = 0.0
    n = len(path)
    for i in range(n):
        u = path[i]
        v = path[(i + 1) % n]
        dx = coords[u][0] - coords[v][0]
        dy = coords[u][1] - coords[v][1]
        dist += math.hypot(dx, dy)
    return dist

def solve_nearest_neighbor(coords):
    """Nearest Neighbor greedy algorithm for TSP."""
    start_time = time.perf_counter()
    n = len(coords)
    if n <= 1:
        return list(range(n)), 0.0, time.perf_counter() - start_time
        
    unvisited = set(range(1, n))
    path = [0]
    curr = 0
    
    while unvisited:
        # Find nearest unvisited node
        next_node = min(
            unvisited,
            key=lambda x: math.hypot(coords[curr][0] - coords[x][0], coords[curr][1] - coords[x][1])
        )
        path.append(next_node)
        unvisited.remove(next_node)
        curr = next_node
        
    dist = calculate_path_distance(path, coords)
    elapsed = time.perf_counter() - start_time
    return path, dist, elapsed

def solve_christofides(coords):
    """NetworkX Christofides 1.5-approximation algorithm."""
    start_time = time.perf_counter()
    n = len(coords)
    if n <= 1:
        return list(range(n)), 0.0, time.perf_counter() - start_time
    if n <= 2:
        return list(range(n)), calculate_path_distance(list(range(n)), coords), time.perf_counter() - start_time
        
    # Build complete graph
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            G.add_edge(i, j, weight=dist)
            
    # Christofides approximation
    raw_path = nx.approximation.traveling_salesman_problem(G, cycle=True)
    
    # Remove duplicate consecutive nodes and drop the returning node for standard representation
    path = []
    seen = set()
    for node in raw_path:
        if node not in seen:
            path.append(node)
            seen.add(node)
            
    # Ensure all nodes are represented (handling edge cases in graph traversal)
    if len(path) < n:
        for i in range(n):
            if i not in seen:
                path.append(i)
                
    dist = calculate_path_distance(path, coords)
    elapsed = time.perf_counter() - start_time
    return path, dist, elapsed

def solve_or_tools(coords, time_limit_seconds=1):
    """Google OR-Tools solver for TSP with local search improvement."""
    start_time = time.perf_counter()
    n = len(coords)
    if n <= 1:
        return list(range(n)), 0.0, time.perf_counter() - start_time
    if n <= 2:
        return list(range(n)), calculate_path_distance(list(range(n)), coords), time.perf_counter() - start_time
        
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        dx = coords[from_node][0] - coords[to_node][0]
        dy = coords[from_node][1] - coords[to_node][1]
        # OR-Tools routing model needs integer distances
        return int(math.hypot(dx, dy) * 1000000)
        
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # Enable Local Search Improvement (Guided Local Search is strong)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit_seconds
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        index = routing.Start(0)
        path = []
        while not routing.IsEnd(index):
            path.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
            
        dist = calculate_path_distance(path, coords)
        elapsed = time.perf_counter() - start_time
        return path, dist, elapsed
    else:
        # Fallback to Nearest Neighbor if OR-Tools fails
        return solve_nearest_neighbor(coords)

def solve_exact(coords):
    """Held-Karp Dynamic Programming Exact TSP solver (N <= 10)."""
    start_time = time.perf_counter()
    n = len(coords)
    if n > 10:
        raise ValueError("Held-Karp exact solver only supported for N <= 10 to prevent hanging.")
    if n <= 1:
        return list(range(n)), 0.0, time.perf_counter() - start_time
    if n <= 2:
        return list(range(n)), calculate_path_distance(list(range(n)), coords), time.perf_counter() - start_time
        
    dist_matrix = [
        [math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) for j in range(n)]
        for i in range(n)
    ]
    
    memo = {}
    
    def get_min_dist(mask, u):
        # Base case: all cities visited, return back to 0
        if mask == (1 << n) - 1:
            return dist_matrix[u][0], 0
            
        key = (mask, u)
        if key in memo:
            return memo[key]
            
        min_dist = float('inf')
        best_next = -1
        
        for v in range(n):
            if not (mask & (1 << v)):
                dist = dist_matrix[u][v] + get_min_dist(mask | (1 << v), v)[0]
                if dist < min_dist:
                    min_dist = dist
                    best_next = v
                    
        memo[key] = (min_dist, best_next)
        return min_dist, best_next
        
    min_dist, next_node = get_min_dist(1, 0)
    
    # Reconstruct path
    path = [0]
    mask = 1
    curr = 0
    while mask < (1 << n) - 1:
        _, next_node = memo[(mask, curr)]
        path.append(next_node)
        mask |= (1 << next_node)
        curr = next_node
        
    elapsed = time.perf_counter() - start_time
    return path, min_dist, elapsed
