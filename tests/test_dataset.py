import unittest
import os
import tempfile
import networkx as nx
from app.solvers.dataset import (
    load_pkl_dataset,
    convert_to_networkx_graph,
    save_pkl_dataset
)

class TestDatasetUtilities(unittest.TestCase):
    def setUp(self):
        # A list of 2 dummy TSP samples of N=4 nodes
        self.coords_list = [
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            [[0.5, 0.5], [0.2, 0.8], [0.9, 0.1], [0.1, 0.4]]
        ]
        self.tours_list = [
            [0, 1, 2, 3],
            [0, 2, 1, 3]
        ]
        
    def test_convert_to_networkx_graph(self):
        coords = self.coords_list[0]
        g = convert_to_networkx_graph(coords)
        
        self.assertIsInstance(g, nx.Graph)
        self.assertEqual(g.number_of_nodes(), 4)
        for i in range(4):
            self.assertEqual(g.nodes[i]['pos'], tuple(coords[i]))
            
    def test_save_and_load_pkl_dataset(self):
        # Create a temp file to test save/load
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            # Save dataset
            save_pkl_dataset(tmp_path, self.coords_list, self.tours_list)
            
            # Load dataset
            loaded_coords, loaded_tours = load_pkl_dataset(tmp_path)
            
            self.assertEqual(len(loaded_coords), 2)
            self.assertEqual(len(loaded_tours), 2)
            
            # Verify coordinates
            for original, loaded in zip(self.coords_list, loaded_coords):
                self.assertEqual(len(loaded), len(original))
                for o_pt, l_pt in zip(original, loaded):
                    self.assertAlmostEqual(o_pt[0], l_pt[0])
                    self.assertAlmostEqual(o_pt[1], l_pt[1])
                    
            # Verify tours (which should be sliced to length N)
            for original, loaded in zip(self.tours_list, loaded_tours):
                self.assertEqual(loaded, original)
                
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
