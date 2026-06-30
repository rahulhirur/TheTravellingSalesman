import pickle
import networkx as nx

def load_pkl_dataset(file_path):
    """
    Loads TSP dataset from a pickle file containing list of (networkx.Graph, tour).
    
    Returns:
        coords_list (list of list of [float, float]): Coordinates normalized in [0, 1].
        tours_list (list of list of int): Optimal tour node sequences sliced to N nodes.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    coords_list = []
    tours_list = []
    
    for item in data:
        g, tour = item
        num_nodes = g.number_of_nodes()
        coords = [g.nodes[i]['pos'] for i in range(num_nodes)]
        coords_list.append(coords)
        # tour has length N+1 (includes returning to start). Slice to first N elements.
        tours_list.append(tour[:num_nodes])
        
    return coords_list, tours_list

def convert_to_networkx_graph(coords):
    """
    Converts list of coordinates to a networkx.Graph with 'pos' attributes on nodes.
    """
    g = nx.Graph()
    for i, c in enumerate(coords):
        g.add_node(i, pos=tuple(c))
    return g

def save_pkl_dataset(file_path, coords_list, tours_list):
    """
    Saves coordinate and tour lists to a pickle file in the standard (networkx.Graph, tour) format.
    """
    data = []
    for coords, tour in zip(coords_list, tours_list):
        g = convert_to_networkx_graph(coords)
        # Ensure the tour ends with the start node for the standard representation (length N+1)
        full_tour = list(tour)
        if len(full_tour) > 0 and full_tour[-1] != full_tour[0]:
            full_tour.append(full_tour[0])
        elif len(full_tour) > 0 and len(full_tour) == len(coords):
            full_tour.append(full_tour[0])
        data.append((g, full_tour))
        
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
