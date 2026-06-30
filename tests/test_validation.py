import unittest
from app.solvers.validation import validate_tsp_coordinates

class TestValidation(unittest.TestCase):
    def test_valid_coordinates(self):
        coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        result = validate_tsp_coordinates(coords)
        self.assertEqual(result, coords)

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            validate_tsp_coordinates("not a list")
            
        with self.assertRaises(ValueError):
            validate_tsp_coordinates([[1, 2], [3]])
            
        with self.assertRaises(ValueError):
            validate_tsp_coordinates([[1, 2], ["a", 3]])

    def test_too_few_nodes(self):
        with self.assertRaises(ValueError):
            validate_tsp_coordinates([[0.0, 0.0], [1.0, 0.0]])

    def test_duplicate_coordinates(self):
        coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
        with self.assertRaises(ValueError):
            validate_tsp_coordinates(coords)

    def test_collinear_coordinates(self):
        coords = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        with self.assertRaises(ValueError):
            validate_tsp_coordinates(coords)

if __name__ == '__main__':
    unittest.main()
