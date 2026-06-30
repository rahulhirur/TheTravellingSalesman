# Next Steps: Running and Deploying Your TSP Solver Platform

Now that the complete project workspace has been generated and structured from scratch, follow these instructions to test, run, and deploy your new MLOps Traveling Salesman solver platform.

---

## 1. Wait for Neural Network Training to Complete
The model training script is currently running in the background to build your Neural TSP Solver:
- It generates 2,000 random 20-city problems.
- Solves them with Google OR-Tools to create high-quality labels.
- Trains a custom PyTorch Pointer Network for 12 epochs.
- Saves the trained model parameters to `app/solvers/model.pth`.

You can check if the file `app/solvers/model.pth` exists. Once it is created, the training is complete.

---

## 2. Verify Your Workspace with the Test Suite
Before starting the servers, verify the correctness of the solvers, validation guards, and API routing by running the unit tests:
```bash
.venv\Scripts\pytest
```
*Note: Make sure your terminal is in the project root folder `d:\Rahul\Projects\TheTravellingSalesman`.*

---

## 3. Run the Backend API Locally
Launch the FastAPI development server:
```bash
.venv\Scripts\uvicorn app.main:app --reload --port 7860
```
- Open `http://localhost:7860/health` in your browser to verify that the server is active, the SQLite database fallback is initialized, and the PyTorch model has loaded successfully.
- You can inspect the interactive OpenAPI documentation at `http://localhost:7860/docs`.

---

## 4. Run the Visualizer Frontend
1. Open the file `frontend/index.html` in your web browser (double-click the file, or right-click and choose a browser).
2. The control panel will automatically check connection health. If the status indicator turns green (**Connected**), you are ready.
3. Click "Generate Random" to place city nodes on the Canvas, or click anywhere on the canvas grid to place nodes manually.
4. Select which solvers you want to execute (Neural, OR-Tools, Christofides, Nearest Neighbor).
5. Click **Solve Traveling Salesman** to fetch the paths.
6. The Canvas will animate the routing steps, and the live comparison charts will benchmark latency and distance performance metrics.
7. Click on any row in the **Run Diagnostics** table to highlight that specific solver's route on the Canvas.

---

## 5. Enable MongoDB Atlas (Optional)
If you want to log solving runs in the cloud:
1. Create a free-tier cluster on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas).
2. Get your connection string (URI).
3. Set the environment variable before starting the API:
   ```powershell
   $env:MONGODB_URI="mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority"
   $env:MONGODB_DB="tsp_solver"
   ```
4. Restart your FastAPI backend. The server health check will now indicate connection to **MongoDB Atlas**. Run metrics will be stored in the cloud, and the **Historical Solver Performance** panel will sync across all users.

---

## 6. Deploy to Hugging Face Spaces (Inference API)
To deploy the backend API for free:
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with **Docker** SDK (choose the Blank template).
2. Under your repository settings in GitHub, add the following Actions Secrets:
   - `HF_TOKEN`: A Hugging Face write token (generated in your Hugging Face account settings).
   - `HF_SPACE_NAME`: Your Space path, e.g., `username/space-name`.
3. Push your code to the `main` branch. GitHub Actions will build the Docker container (automatically training the model inside it) and deploy it to Hugging Face.

---

## 7. Deploy to GitHub Pages (Static Frontend)
To host the frontend visualizer:
1. Go to your GitHub repository **Settings > Pages**.
2. Select **Deploy from a branch**, choose `main` branch, and set the folder to `/frontend`.
3. Save. Once deployed, open the page, click the Settings button, and change the **API Endpoint URL** to your Hugging Face Space URL (e.g. `https://username-space-name.hf.space`).
