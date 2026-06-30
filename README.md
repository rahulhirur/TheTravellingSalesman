# Traveling Salesman Problem Solver: Neural vs. Conventional Benchmarking

An end-to-end interactive benchmarking platform and MLOps pipeline designed to compare Deep Learning approaches (PyTorch-based Pointer Networks) against Conventional Algorithmic solvers (Google OR-Tools, NetworkX Christofides 1.5-approximation, Nearest Neighbor, and Held-Karp exact solver) on the Traveling Salesman Problem (TSP).

## Tech Stack
* **Frontend**: HTML5 Canvas, Vanilla CSS (Premium Glassmorphism Dark Theme), Javascript
* **Backend API**: Python, FastAPI, Uvicorn
* **ML Engine**: PyTorch (Pointer Network utilizing Multi-Head Self-Attention)
* **Conventional Solvers**: SciPy, NetworkX, Google OR-Tools
* **Database / Logs**: MongoDB Atlas (Primary) with SQLite (Local Offline Fallback)
* **Package Management**: `uv`
* **CI/CD / Containerization**: GitHub Actions & Docker (Configured for Hugging Face Spaces)

---

## Directory Structure
```text
TheTravellingSalesman/
├── .github/workflows/deploy.yml   # CI/CD deployment to Hugging Face Spaces
├── app/
│   ├── main.py                    # FastAPI application and DB logging
│   └── solvers/
│       ├── __init__.py            # Export solver interfaces
│       ├── conventional.py        # Conventional solvers (OR-Tools, Christofides, NN, Exact)
│       └── neural.py              # PyTorch Pointer Network inference wrapper
├── scripts/
│   └── train.py                   # Data generation & Pointer Network training pipeline
├── frontend/
│   ├── index.html                 # Main interface structural markup
│   ├── style.css                  # Premium dark theme styling
│   └── app.js                     # Canvas drawing, metrics, and API caller
├── tests/
│   ├── test_solvers.py            # Unit tests for routing solver logic
│   └── test_api.py                # Integration tests for FastAPI endpoints
├── pyproject.toml                 # Dependencies and build settings
└── Dockerfile                     # API containerization with baked-in training phase
```

---

## Setup & Running Locally

This project uses **`uv`** for rapid dependency resolution.

### 1. Initialize Virtual Environment & Install Dependencies
Ensure you have `uv` installed. If not, install it via:
```bash
pip install uv
```

Then synchronize the dependencies:
```bash
uv sync
```
This automatically downloads Python and sets up the virtual environment in `.venv/` with all packages in `pyproject.toml`.

### 2. Train the Neural Solver
Run the training script to generate synthetic TSP coordinates, solve them with OR-Tools, train the neural Pointer Network, and save the model weights to `app/solvers/model.pth`:
```bash
.venv\Scripts\python.exe scripts/train.py
```

### 3. Run the Backend API
Start the FastAPI server locally on port 7860 (default):
```bash
.venv\Scripts\uvicorn app.main:app --reload --port 7860
```

### 4. Launch the Frontend UI
Simply open `frontend/index.html` in any web browser. You can click to add cities, generate random ones, and benchmark the solvers in real time.

### 5. Running Tests
Run the test suite using pytest to verify solvers and API endpoints:
```bash
.venv\Scripts\pytest
```

---

## Environment Variables
* `MONGODB_URI`: Connection string for MongoDB Atlas database. If not set, the application automatically falls back to logging runs in a local SQLite database (`app/runs.db`).
* `MONGODB_DB`: Name of the database to use (defaults to `tsp_solver`).
