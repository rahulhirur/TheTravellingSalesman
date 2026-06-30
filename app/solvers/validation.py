import math

def validate_tsp_coordinates(points, max_nodes=50):
    """
    Validates a list of 2D coordinates for TSP solving.
    Returns the points if valid. Raises ValueError for invalid configurations.
    """
    if not isinstance(points, list):
        raise ValueError("Points must be a list of coordinates.")
        
    n = len(points)
    if n < 3:
        raise ValueError(f"TSP requires at least 3 nodes to form a tour. Found {n}.")
        
    if n > max_nodes:
        raise ValueError(f"Max supported nodes is {max_nodes}. Found {n}.")
        
    seen = set()
    for idx, p in enumerate(points):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"Point at index {idx} must be a 2D coordinate [x, y].")
            
        try:
            x, y = float(p[0]), float(p[1])
        except (ValueError, TypeError):
            raise ValueError(f"Point at index {idx} must contain numeric values.")
            
        # Check duplicates with small tolerance (1e-6)
        key = (round(x, 6), round(y, 6))
        if key in seen:
            raise ValueError(f"Duplicate coordinate detected at index {idx}: {p}.")
        seen.add(key)

    # Check collinearity: if all points lie on a straight line, TSP is degenerate
    if n >= 3:
        x0, y0 = float(points[0][0]), float(points[0][1])
        x1, y1 = float(points[1][0]), float(points[1][1])
        dx1 = x1 - x0
        dy1 = y1 - y0
        
        collinear = True
        for i in range(2, n):
            xi, yi = float(points[i][0]), float(points[i][1])
            dxi = xi - x0
            dyi = yi - y0
            # Cross product
            cross_product = dx1 * dyi - dy1 * dxi
            if abs(cross_product) > 1e-6:
                collinear = False
                break
                
        if collinear:
            raise ValueError("All coordinates are collinear (lie on a straight line).")
            
    return points
