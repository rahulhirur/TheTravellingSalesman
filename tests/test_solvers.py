import unittest
import math
import numpy as np
from app.solvers.conventional import (
    solve_nearest_neighbor,
    solve_christofides,
    solve_or_tools,
    solve_exact,
    calculate_path_distance
)
from app.solvers.notebook_transformer import solve_notebook_transformer

class TestSolvers(unittest.TestCase):
    def setUp(self):
        # Create a simple test graph (a unit square with an extra point)
        # 5 points
        self.coords_5 = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5]
        ]
        # 12 points (exact solver should fail/raise error since N > 10)
        self.coords_12 = np.random.rand(12, 2).tolist()
        # 20 points for notebook transformer testing
        self.coords_20 = np.random.rand(20, 2).tolist()

    def test_calculate_path_distance(self):
        # A simple triangle path: 0 -> 1 -> 2 -> (0)
        # coords: (0,0), (0,3), (4,0)
        # distance: 3 + 5 + 4 = 12
        coords = [[0.0, 0.0], [0.0, 3.0], [4.0, 0.0]]
        path = [0, 1, 2]
        dist = calculate_path_distance(path, coords)
        self.assertAlmostEqual(dist, 12.0)

    def test_nearest_neighbor(self):
        path, dist, elapsed = solve_nearest_neighbor(self.coords_5)
        self.assertEqual(len(path), 5)
        self.assertEqual(set(path), set(range(5)))
        self.assertEqual(path[0], 0)
        self.assertTrue(dist > 0)
        self.assertTrue(elapsed >= 0)

    def test_christofides(self):
        path, dist, elapsed = solve_christofides(self.coords_5)
        self.assertEqual(len(path), 5)
        self.assertEqual(set(path), set(range(5)))
        self.assertEqual(path[0], 0)
        self.assertTrue(dist > 0)
        self.assertTrue(elapsed >= 0)

    def test_or_tools(self):
        path, dist, elapsed = solve_or_tools(self.coords_5)
        self.assertEqual(len(path), 5)
        self.assertEqual(set(path), set(range(5)))
        self.assertEqual(path[0], 0)
        self.assertTrue(dist > 0)
        self.assertTrue(elapsed >= 0)

    def test_exact_held_karp(self):
        # Test exact solver on N=5
        path, dist, elapsed = solve_exact(self.coords_5)
        self.assertEqual(len(path), 5)
        self.assertEqual(set(path), set(range(5)))
        self.assertEqual(path[0], 0)
        
        # Exact solver must find the absolute minimum path
        # Compare distance with nearest neighbor (which might be equal or worse)
        _, nn_dist, _ = solve_nearest_neighbor(self.coords_5)
        self.assertTrue(dist <= nn_dist + 1e-6)

        # Test exact solver raises error for N > 10
        with self.assertRaises(ValueError):
            solve_exact(self.coords_12)

    def test_notebook_transformer_fallback(self):
        # N=5 != 20, should fallback to nearest neighbor
        path, dist, elapsed = solve_notebook_transformer(self.coords_5)
        self.assertEqual(len(path), 5) # nearest neighbor returns path of length N
        self.assertEqual(set(path), set(range(5)))

    def test_notebook_transformer_solve(self):
        # N=20, should run model forward pass and return path
        path, dist, elapsed = solve_notebook_transformer(self.coords_20, use_ga=False)
        self.assertEqual(len(path), 21) # N+1 for completed loop
        self.assertEqual(set(path), set(range(20)))
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 0)
        self.assertTrue(dist > 0)

if __name__ == '__main__':
    unittest.main()
