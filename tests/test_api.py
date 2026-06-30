import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "notebook_model_loaded" in data

def test_solve_endpoint():
    payload = {
        "points": [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5]
        ],
        "solvers": ["nearest_neighbor", "or_tools"]
    }
    response = client.post("/solve", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "results" in data
    
    results = data["results"]
    for solver in ["nearest_neighbor", "or_tools"]:
        assert solver in results
        assert "path" in results[solver]
        assert "distance" in results[solver]
        assert "time_taken" in results[solver]
        
        path = results[solver]["path"]
        assert len(path) == 5
        assert set(path) == set(range(5))

def test_solve_endpoint_validation():
    # Empty points list
    payload = {
        "points": [],
        "solvers": ["nearest_neighbor"]
    }
    response = client.post("/solve", json=payload)
    assert response.status_code == 400

    # Exact solver N > 10 restriction
    payload = {
        "points": [[0.1 * i, 0.1 * (i % 2)] for i in range(12)],
        "solvers": ["exact"]
    }
    response = client.post("/solve", json=payload)
    assert response.status_code == 400
    assert "Exact Solver only supports N <= 10" in response.json()["detail"]

def test_solve_endpoint_duplicate_validation():
    # Duplicate points
    payload = {
        "points": [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        "solvers": ["nearest_neighbor"]
    }
    response = client.post("/solve", json=payload)
    assert response.status_code == 400
    assert "Duplicate coordinate detected" in response.json()["detail"]

def test_solve_endpoint_collinear_validation():
    # Collinear points
    payload = {
        "points": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        "solvers": ["nearest_neighbor"]
    }
    response = client.post("/solve", json=payload)
    assert response.status_code == 400
    assert "collinear" in response.json()["detail"]
