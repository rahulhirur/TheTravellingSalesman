// TSP Solver Frontend Application Logic

// Configuration & State
const state = {
    cities: [], // Array of {x, y} coordinates (normalized coordinates in canvas logical space 0-1000)
    results: {}, // Solver results from backend
    selectedSolverKey: null, // Key of the solver currently highlighted on the canvas
    isSolving: false,
    animation: {
        active: false,
        startTime: null,
        duration: 1500, // duration in ms
        solverKey: null
    },
    apiBaseUrl: 'http://localhost:7860',
    colorPalette: {
        neural: '#06b6d4',       // Neon Cyan
        or_tools: '#3b82f6',     // Neon Blue
        christofides: '#8b5cf6', // Neon Purple
        nearest_neighbor: '#f97316', // Neon Orange
        exact: '#10b981'         // Neon Green
    },
    solverNames: {
        neural: 'Neural Solver (Pointer Net)',
        or_tools: 'Google OR-Tools Heuristic',
        christofides: 'Christofides Approximation',
        nearest_neighbor: 'Nearest Neighbor Heuristic',
        exact: 'Held-Karp Exact Solver'
    },
    theme: 'light'
};

// Canvas Constants
const CANVAS_LOGICAL_SIZE = 1000;
let canvas, ctx;
let animationFrameId = null;

// DOM Elements
const elements = {
    apiUrlInput: document.getElementById('api-url-input'),
    apiTestBtn: document.getElementById('api-test-btn'),
    connectionStatus: document.getElementById('connection-status'),
    citiesCountSlider: document.getElementById('cities-count-slider'),
    citiesCountVal: document.getElementById('cities-count-val'),
    generateRandomBtn: document.getElementById('generate-random-btn'),
    clearCanvasBtn: document.getElementById('clear-canvas-btn'),
    solverNeuralCheck: document.getElementById('solver-neural-check'),
    solverOrtoolsCheck: document.getElementById('solver-ortools-check'),
    solverChristofidesCheck: document.getElementById('solver-christofides-check'),
    solverGreedyCheck: document.getElementById('solver-greedy-check'),
    solverExactCheck: document.getElementById('solver-exact-check'),
    exactSolverWrapper: document.getElementById('exact-solver-wrapper'),
    animationSpeedSlider: document.getElementById('animation-speed-slider'),
    animationSpeedVal: document.getElementById('animation-speed-val'),
    solveBtn: document.getElementById('solve-btn'),
    resetViewBtn: document.getElementById('reset-view-btn'),
    activeNodeCount: document.getElementById('active-node-count'),
    canvasPrompt: document.getElementById('canvas-prompt'),
    dbStatus: document.getElementById('db-status'),
    metricsTableBody: document.getElementById('metrics-table-body'),
    distanceChart: document.getElementById('distance-chart'),
    latencyChart: document.getElementById('latency-chart'),
    sysBackend: document.getElementById('mlops-backend-status'),
    sysDb: document.getElementById('mlops-db-status'),
    sysModel: document.getElementById('mlops-model-status'),
    refreshMlopsBtn: document.getElementById('refresh-mlops-btn'),
    historicalAggregatesContainer: document.getElementById('historical-aggregates-container'),
    
    // MLOps Dashboard Elements
    telemetryCountVal: document.getElementById('telemetry-count-val'),
    downloadDatasetBtn: document.getElementById('download-dataset-btn'),
    retrainStatusText: document.getElementById('retrain-status-text'),
    retrainProgressWrapper: document.getElementById('retrain-progress-wrapper'),
    retrainProgressBar: document.getElementById('retrain-progress-bar'),
    triggerRetrainBtn: document.getElementById('trigger-retrain-btn'),
    retrainConsoleLog: document.getElementById('retrain-console-log'),
    themeToggleBtn: document.getElementById('theme-toggle-btn')
};

// Initialize App
window.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    setupEventListeners();
    initArchitectureVisualizer();
    
    // Load saved API URL from localStorage if available
    const savedUrl = localStorage.getItem('tsp_api_url');
    if (savedUrl) {
        state.apiBaseUrl = savedUrl;
        elements.apiUrlInput.value = savedUrl;
    }
    
    // Load theme preference or default to light
    const savedTheme = localStorage.getItem('tsp_theme') || 'light';
    setTheme(savedTheme);
    
    checkServerHealth();
});

// Canvas Setup with Retina Display Support
function initCanvas() {
    canvas = document.getElementById('tsp-canvas');
    ctx = canvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

function resizeCanvas() {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    
    ctx.scale(dpr, dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    
    draw();
}

// Draw Canvas elements
function draw() {
    if (!canvas || !ctx) return;
    
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    
    ctx.clearRect(0, 0, width, height);
    
    // 1. Draw Grid Pattern
    const isDark = state.theme === 'dark';
    ctx.strokeStyle = isDark ? 'rgba(255, 255, 255, 0.03)' : 'rgba(15, 23, 42, 0.05)';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }
    
    // 2. Draw Paths
    if (state.animation.active && state.animation.solverKey) {
        // Draw animated route
        drawRouteAnimated(state.animation.solverKey);
    } else {
        // Draw static paths
        drawPathsStatic();
    }
    
    // 3. Draw Nodes (Cities)
    state.cities.forEach((city, index) => {
        const screenPos = toScreenCoords(city.x, city.y);
        
        // Node halo/shadow
        ctx.beginPath();
        ctx.arc(screenPos.x, screenPos.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(15, 23, 42, 0.04)';
        ctx.fill();
        
        // Node center
        ctx.beginPath();
        ctx.arc(screenPos.x, screenPos.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Node index text (only draw for N <= 40 to avoid clutter)
        if (state.cities.length <= 40) {
            ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.5)' : 'rgba(15, 23, 42, 0.6)';
            ctx.font = '10px "Outfit", sans-serif';
            ctx.fillText(index.toString(), screenPos.x + 8, screenPos.y - 4);
        }
    });
}

// Convert normalized logic coordinates [0, 1000] to Screen Coordinates
function toScreenCoords(lx, ly) {
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    
    // Maintain square aspect ratio inside visualizer
    const padding = 30;
    const renderSize = Math.min(width, height) - (padding * 2);
    
    const xOffset = (width - renderSize) / 2;
    const yOffset = (height - renderSize) / 2;
    
    return {
        x: xOffset + (lx / CANVAS_LOGICAL_SIZE) * renderSize,
        y: yOffset + (ly / CANVAS_LOGICAL_SIZE) * renderSize
    };
}

// Convert Screen Coordinates back to Logical [0, 1000] coordinates
function toLogicalCoords(sx, sy) {
    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    
    const padding = 30;
    const renderSize = Math.min(width, height) - (padding * 2);
    
    const xOffset = (width - renderSize) / 2;
    const yOffset = (height - renderSize) / 2;
    
    const lx = ((sx - xOffset) / renderSize) * CANVAS_LOGICAL_SIZE;
    const ly = ((sy - yOffset) / renderSize) * CANVAS_LOGICAL_SIZE;
    
    return {
        x: Math.max(0, Math.min(CANVAS_LOGICAL_SIZE, lx)),
        y: Math.max(0, Math.min(CANVAS_LOGICAL_SIZE, ly))
    };
}

// Draw static paths
function drawPathsStatic() {
    Object.keys(state.results).forEach(key => {
        const routeData = state.results[key];
        if (!routeData || !routeData.path) return;
        
        const isHighlighted = (state.selectedSolverKey === key);
        const hasSelection = (state.selectedSolverKey !== null);
        
        ctx.beginPath();
        const path = routeData.path;
        
        // Setup line style
        ctx.strokeStyle = state.colorPalette[key];
        ctx.lineWidth = isHighlighted ? 3 : 1.5;
        
        if (hasSelection) {
            ctx.globalAlpha = isHighlighted ? 1.0 : 0.1;
        } else {
            ctx.globalAlpha = 0.6; // Show all partially when none is selected
        }
        
        // Path trace
        for (let i = 0; i <= path.length; i++) {
            const nodeIdx = path[i % path.length];
            const city = state.cities[nodeIdx];
            const screenPos = toScreenCoords(city.x, city.y);
            if (i === 0) {
                ctx.moveTo(screenPos.x, screenPos.y);
            } else {
                ctx.lineTo(screenPos.x, screenPos.y);
            }
        }
        
        // Add shadow glowing effect to the highlighted path
        if (isHighlighted) {
            ctx.shadowColor = state.colorPalette[key];
            ctx.shadowBlur = 10;
        }
        
        ctx.stroke();
        
        // Clear effects for subsequent draws
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1.0;
    });
}

// Draw path animated
function drawRouteAnimated(solverKey) {
    const routeData = state.results[solverKey];
    if (!routeData || !routeData.path) return;
    
    const path = routeData.path;
    const totalLegs = path.length; // includes return leg
    
    const elapsed = performance.now() - state.animation.startTime;
    const progress = Math.min(1, elapsed / state.animation.duration);
    
    // Draw paths up to current progress
    const legsToDraw = progress * totalLegs;
    const fullLegs = Math.floor(legsToDraw);
    const partialLegPercent = legsToDraw - fullLegs;
    
    ctx.strokeStyle = state.colorPalette[solverKey];
    ctx.lineWidth = 3.5;
    ctx.shadowColor = state.colorPalette[solverKey];
    ctx.shadowBlur = 12;
    
    ctx.beginPath();
    
    // 1. Draw fully completed legs
    for (let i = 0; i <= fullLegs; i++) {
        const nodeIdx = path[i % path.length];
        const city = state.cities[nodeIdx];
        const screenPos = toScreenCoords(city.x, city.y);
        
        if (i === 0) {
            ctx.moveTo(screenPos.x, screenPos.y);
        } else {
            ctx.lineTo(screenPos.x, screenPos.y);
        }
    }
    
    // 2. Draw partial leg
    if (fullLegs < totalLegs && partialLegPercent > 0) {
        const currNodeIdx = path[fullLegs % path.length];
        const nextNodeIdx = path[(fullLegs + 1) % path.length];
        
        const currCity = state.cities[currNodeIdx];
        const nextCity = state.cities[nextNodeIdx];
        
        const currScreen = toScreenCoords(currCity.x, currCity.y);
        const nextScreen = toScreenCoords(nextCity.x, nextCity.y);
        
        const targetX = currScreen.x + (nextScreen.x - currScreen.x) * partialLegPercent;
        const targetY = currScreen.y + (nextScreen.y - currScreen.y) * partialLegPercent;
        
        ctx.lineTo(targetX, targetY);
    }
    
    ctx.stroke();
    
    // Clear styles
    ctx.shadowBlur = 0;
    
    if (progress < 1) {
        animationFrameId = requestAnimationFrame(() => draw());
    } else {
        state.animation.active = false;
        state.animation.solverKey = null;
        draw(); // draw final static paths
    }
}

// Generate uniform random nodes
function generateRandomCities(count) {
    stopAnimation();
    state.cities = [];
    state.results = {};
    state.selectedSolverKey = null;
    
    // Keep padding to avoid cities right on the edge
    const pad = 80;
    for (let i = 0; i < count; i++) {
        state.cities.push({
            x: pad + Math.random() * (CANVAS_LOGICAL_SIZE - pad * 2),
            y: pad + Math.random() * (CANVAS_LOGICAL_SIZE - pad * 2)
        });
    }
    
    updateUI();
}

function clearCanvas() {
    stopAnimation();
    state.cities = [];
    state.results = {};
    state.selectedSolverKey = null;
    updateUI();
}

function stopAnimation() {
    state.animation.active = false;
    state.animation.solverKey = null;
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// Event Listeners
function setupEventListeners() {
    // Canvas Click Listener (Place City)
    canvas.addEventListener('click', (e) => {
        if (state.isSolving) return;
        stopAnimation();
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const logical = toLogicalCoords(x, y);
        state.cities.push(logical);
        
        // Reset results since cities changed
        state.results = {};
        state.selectedSolverKey = null;
        
        updateUI();
    });
    
    // Test API Connection
    elements.apiTestBtn.addEventListener('click', () => {
        const url = elements.apiUrlInput.value.trim().replace(/\/$/, "");
        state.apiBaseUrl = url;
        localStorage.setItem('tsp_api_url', url);
        checkServerHealth();
    });
    
    // Cities slider change
    elements.citiesCountSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        elements.citiesCountVal.textContent = val;
        
        // Exact solver selection gating
        if (parseInt(val) > 10) {
            elements.solverExactCheck.checked = false;
            elements.solverExactCheck.disabled = true;
            elements.exactSolverWrapper.classList.add('disabled');
        } else {
            elements.solverExactCheck.disabled = false;
            elements.exactSolverWrapper.classList.remove('disabled');
        }
    });
    
    // Animation Speed slider change
    elements.animationSpeedSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        state.animation.duration = parseInt(val);
        elements.animationSpeedVal.textContent = val === '0' ? 'Instant' : `${(val / 1000).toFixed(1)}s`;
    });
    
    // Buttons
    elements.generateRandomBtn.addEventListener('click', () => {
        const count = parseInt(elements.citiesCountSlider.value);
        generateRandomCities(count);
    });
    
    elements.clearCanvasBtn.addEventListener('click', clearCanvas);
    
    elements.resetViewBtn.addEventListener('click', () => {
        draw();
    });
    
    elements.solveBtn.addEventListener('click', solveTSP);
    
    elements.refreshMlopsBtn.addEventListener('click', () => {
        fetchHistoricalMetrics();
        fetchTelemetryStats();
    });

    // Tabs Navigation Routing
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            // Toggle active button
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle active panel
            document.querySelectorAll('.tab-panel').forEach(panel => {
                panel.classList.remove('active-panel');
            });
            document.getElementById(tabId).classList.add('active-panel');
            
            // If MLOps Dashboard is selected, refresh stats
            if (tabId === 'mlops-tab') {
                fetchTelemetryStats();
            }
        });
    });

    // Telemetry Download Dataset
    elements.downloadDatasetBtn.addEventListener('click', () => {
        if (!telemetryData || telemetryData.length === 0) return;
        
        const jsonStr = JSON.stringify(telemetryData, null, 2);
        const blob = new Blob([jsonStr], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `tsp_telemetry_dataset_${telemetryData.length}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    });

    // Model Retraining trigger
    elements.triggerRetrainBtn.addEventListener('click', triggerRetraining);
    
    // Theme Toggle listener
    if (elements.themeToggleBtn) {
        elements.themeToggleBtn.addEventListener('click', () => {
            const nextTheme = state.theme === 'light' ? 'dark' : 'light';
            setTheme(nextTheme);
        });
    }
}

// Global theme control
function setTheme(theme) {
    state.theme = theme;
    localStorage.setItem('tsp_theme', theme);
    
    const body = document.body;
    const toggleBtn = elements.themeToggleBtn;
    
    if (theme === 'dark') {
        body.classList.add('dark-theme');
        if (toggleBtn) toggleBtn.textContent = '☀️';
    } else {
        body.classList.remove('dark-theme');
        if (toggleBtn) toggleBtn.textContent = '🌙';
    }
    
    // Redraw the canvas with the new theme colors
    draw();
    
    // Redraw the architecture diagram SVG with the new theme colors
    if (typeof renderArchitectureDiagram === 'function') {
        renderArchitectureDiagram();
    }
}

// Update local UI states
function updateUI() {
    const count = state.cities.length;
    elements.activeNodeCount.textContent = count;
    
    if (count > 0) {
        elements.canvasPrompt.style.opacity = 0;
    } else {
        elements.canvasPrompt.style.opacity = 1;
    }
    
    draw();
}

// Server Health Verification
async function checkServerHealth() {
    setConnectionStatus('warning', 'Connecting...');
    try {
        const response = await fetch(`${state.apiBaseUrl}/health`);
        if (!response.ok) throw new Error('Non-200 response');
        const data = await response.json();
        
        setConnectionStatus('online', 'Connected');
        elements.sysBackend.textContent = 'Online';
        elements.sysBackend.className = 'badge success';
        
        elements.sysDb.textContent = data.database_type === 'mongodb' ? 'MongoDB Atlas' : 'SQLite Local';
        elements.sysDb.className = 'badge info';
        
        elements.sysModel.textContent = data.neural_model_loaded ? 'Loaded' : 'Fallback Mode';
        elements.sysModel.className = data.neural_model_loaded ? 'badge success' : 'badge offline';
        
        fetchHistoricalMetrics();
        
        // Check if retraining is active
        const statusResponse = await fetch(`${state.apiBaseUrl}/train/status`);
        const statusData = await statusResponse.json();
        if (statusData.is_training) {
            updateRetrainUI(true, 'Training...');
            startRetrainPolling();
        } else {
            updateRetrainUI(false, 'Idle');
        }
    } catch (err) {
        setConnectionStatus('offline', 'Offline');
        elements.sysBackend.textContent = 'Offline';
        elements.sysBackend.className = 'badge offline';
        elements.sysDb.textContent = 'None';
        elements.sysDb.className = 'badge offline';
        elements.sysModel.textContent = 'No';
        elements.sysModel.className = 'badge offline';
        updateRetrainUI(false, 'Idle');
    }
}

function setConnectionStatus(status, text) {
    const dot = elements.connectionStatus.querySelector('.status-dot');
    const label = elements.connectionStatus.querySelector('.status-text');
    
    dot.className = `status-dot ${status}`;
    label.textContent = text;
}

// Post request to solve TSP
async function solveTSP() {
    if (state.cities.length < 3) {
        alert("Please place at least 3 cities on the canvas to solve TSP.");
        return;
    }
    
    // Gather selected solvers
    const selectedSolvers = [];
    if (elements.solverNeuralCheck.checked) selectedSolvers.push('neural');
    if (elements.solverOrtoolsCheck.checked) selectedSolvers.push('or_tools');
    if (elements.solverChristofidesCheck.checked) selectedSolvers.push('christofides');
    if (elements.solverGreedyCheck.checked) selectedSolvers.push('nearest_neighbor');
    if (elements.solverExactCheck.checked && state.cities.length <= 10) selectedSolvers.push('exact');
    
    if (selectedSolvers.length === 0) {
        alert("Please select at least one solver algorithm.");
        return;
    }
    
    state.isSolving = true;
    elements.solveBtn.disabled = true;
    elements.solveBtn.textContent = 'Computing routes...';
    
    // Normalize coordinates representation to [0.0, 1.0] relative bounding box for solver API
    const normalizedPoints = state.cities.map(c => [c.x / CANVAS_LOGICAL_SIZE, c.y / CANVAS_LOGICAL_SIZE]);
    
    try {
        const response = await fetch(`${state.apiBaseUrl}/solve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                points: normalizedPoints,
                solvers: selectedSolvers
            })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'API solve request failed');
        }
        
        const data = await response.json();
        state.results = data.results;
        
        // Update Log status
        if (data.logged) {
            elements.dbStatus.textContent = 'Run Logged';
            elements.dbStatus.className = 'db-status-pill success';
        } else {
            elements.dbStatus.textContent = 'Run Not Logged';
            elements.dbStatus.className = 'db-status-pill';
        }
        
        renderMetrics();
        
        // Start animation for the primary selected solver
        const primarySolver = selectedSolvers.includes('neural') ? 'neural' : selectedSolvers[0];
        highlightSolver(primarySolver);
        
        // Trigger aggregates refresh
        fetchHistoricalMetrics();
        
    } catch (err) {
        alert(`Error solving TSP: ${err.message}`);
    } finally {
        state.isSolving = false;
        elements.solveBtn.disabled = false;
        elements.solveBtn.textContent = 'Solve Traveling Salesman';
    }
}

// Display metrics in table and charts
function renderMetrics() {
    const tbody = elements.metricsTableBody;
    tbody.innerHTML = '';
    
    // Find best solvers (minimum distance, minimum latency)
    let bestDist = floatInfinity();
    let bestTime = floatInfinity();
    let bestDistSolver = null;
    let bestTimeSolver = null;
    
    Object.keys(state.results).forEach(key => {
        const res = state.results[key];
        if (res.distance < bestDist) {
            bestDist = res.distance;
            bestDistSolver = key;
        }
        if (res.time_taken < bestTime) {
            bestTime = res.time_taken;
            bestTimeSolver = key;
        }
    });
    
    // Build table rows
    Object.keys(state.results).forEach(key => {
        const res = state.results[key];
        const isBestDist = (key === bestDistSolver);
        const isBestTime = (key === bestTimeSolver);
        
        // Compute accuracy gap compared to best distance (expressed as percentage)
        let gapText = '0.00%';
        if (!isBestDist && bestDist > 0) {
            const gap = ((res.distance - bestDist) / bestDist) * 100;
            gapText = `+${gap.toFixed(2)}%`;
        }
        
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        if (state.selectedSolverKey === key) tr.classList.add('selected-row');
        
        // Highlight row click
        tr.addEventListener('click', () => {
            highlightSolver(key);
            // Highlight selected row in table visual
            document.querySelectorAll('.metrics-table tbody tr').forEach(r => r.classList.remove('selected-row'));
            tr.classList.add('selected-row');
        });
        
        tr.innerHTML = `
            <td>
                <div class="solver-name-col">
                    <span class="color-indicator" style="background-color: ${state.colorPalette[key]}"></span>
                    <span>${state.solverNames[key]}</span>
                </div>
            </td>
            <td>${res.distance.toFixed(4)}</td>
            <td>${(res.time_taken * 1000).toFixed(2)} ms</td>
            <td class="accuracy-gap-val">${gapText}</td>
            <td>
                <span class="badge ${isBestDist ? 'best' : ''}">
                    ${isBestDist ? 'Shortest' : 'Suboptimal'}
                </span>
            </td>
        `;
        tbody.appendChild(tr);
    });
    
    // Draw charts
    renderCharts(bestDist, bestTime);
}

function floatInfinity() {
    return 99999999;
}

function highlightSolver(solverKey) {
    stopAnimation();
    state.selectedSolverKey = solverKey;
    
    const duration = state.animation.duration;
    if (duration > 0) {
        state.animation.active = true;
        state.animation.startTime = performance.now();
        state.animation.solverKey = solverKey;
    }
    
    draw();
}

// Custom SVG/CSS Bar Charts Drawing
function renderCharts(bestDist, bestTime) {
    // 1. Distance Chart
    const distContainer = elements.distanceChart;
    distContainer.innerHTML = '';
    
    // Find maximum distance to normalize bar widths
    let maxDist = 0;
    Object.keys(state.results).forEach(key => {
        if (state.results[key].distance > maxDist) maxDist = state.results[key].distance;
    });
    
    Object.keys(state.results).forEach(key => {
        const val = state.results[key].distance;
        const pct = maxDist > 0 ? (val / maxDist) * 100 : 0;
        
        const item = document.createElement('div');
        item.className = 'chart-bar-item';
        item.innerHTML = `
            <div class="bar-label-row">
                <span class="bar-label">${key === 'neural' ? 'Neural' : key.toUpperCase()}</span>
                <span class="bar-value">${val.toFixed(4)}</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${pct}%; background-color: ${state.colorPalette[key]}"></div>
            </div>
        `;
        distContainer.appendChild(item);
    });
    
    // 2. Latency Chart
    const latencyContainer = elements.latencyChart;
    latencyContainer.innerHTML = '';
    
    let maxTime = 0;
    Object.keys(state.results).forEach(key => {
        if (state.results[key].time_taken > maxTime) maxTime = state.results[key].time_taken;
    });
    
    Object.keys(state.results).forEach(key => {
        const val = state.results[key].time_taken * 1000; // to ms
        const maxValMs = maxTime * 1000;
        const pct = maxValMs > 0 ? (val / maxValMs) * 100 : 0;
        
        const item = document.createElement('div');
        item.className = 'chart-bar-item';
        item.innerHTML = `
            <div class="bar-label-row">
                <span class="bar-label">${key === 'neural' ? 'Neural' : key.toUpperCase()}</span>
                <span class="bar-value">${val.toFixed(2)} ms</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${pct}%; background-color: ${state.colorPalette[key]}"></div>
            </div>
        `;
        latencyContainer.appendChild(item);
    });
}

// Fetch aggregates from database
async function fetchHistoricalMetrics() {
    const container = elements.historicalAggregatesContainer;
    try {
        const response = await fetch(`${state.apiBaseUrl}/metrics`);
        if (!response.ok) throw new Error('API metrics query failed');
        const data = await response.json();
        
        if (Object.keys(data).length === 0) {
            container.innerHTML = '<div class="chart-placeholder">No historical run records found in database. Solve some TSPs to start gathering telemetry.</div>';
            return;
        }
        
        container.innerHTML = '';
        
        // Find max values for normalization
        let maxDist = 0;
        let maxTime = 0;
        Object.keys(data).forEach(key => {
            if (data[key].avg_distance > maxDist) maxDist = data[key].avg_distance;
            if (data[key].avg_time_taken > maxTime) maxTime = data[key].avg_time_taken;
        });
        
        // Create clean breakdown layout
        const listWrapper = document.createElement('div');
        listWrapper.style.display = 'flex';
        listWrapper.style.flexDirection = 'column';
        listWrapper.style.gap = '14px';
        
        Object.keys(data).forEach(key => {
            const avgDist = data[key].avg_distance;
            const avgTime = data[key].avg_time_taken;
            const count = data[key].count;
            
            const distPct = maxDist > 0 ? (avgDist / maxDist) * 100 : 0;
            const timePct = maxTime > 0 ? (avgTime / maxTime) * 100 : 0;
            
            const card = document.createElement('div');
            card.style.background = 'rgba(255, 255, 255, 0.01)';
            card.style.border = '1px solid rgba(255, 255, 255, 0.03)';
            card.style.borderRadius = '6px';
            card.style.padding = '10px 14px';
            
            card.innerHTML = `
                <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: ${state.colorPalette[key] || '#ffffff'}">
                    <span>${state.solverNames[key] || key}</span>
                    <span style="font-size: 11px; font-weight: normal; color: var(--text-muted)">(${count} runs)</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <!-- Avg Distance -->
                    <div style="display: flex; flex-direction: column; gap: 2px;">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary)">
                            <span>Avg Distance</span>
                            <span style="font-family: var(--font-mono)">${avgDist.toFixed(4)}</span>
                        </div>
                        <div style="height: 4px; background: rgba(255, 255, 255, 0.03); border-radius: 2px; overflow: hidden;">
                            <div style="height: 100%; width: ${distPct}%; background: ${state.colorPalette[key] || '#ffffff'}; border-radius: 2px;"></div>
                        </div>
                    </div>
                    <!-- Avg Latency -->
                    <div style="display: flex; flex-direction: column; gap: 2px;">
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary)">
                            <span>Avg Latency</span>
                            <span style="font-family: var(--font-mono)">${avgTime.toFixed(2)} ms</span>
                        </div>
                        <div style="height: 4px; background: rgba(255, 255, 255, 0.03); border-radius: 2px; overflow: hidden;">
                            <div style="height: 100%; width: ${timePct}%; background: ${state.colorPalette[key] || '#ffffff'}; border-radius: 2px;"></div>
                        </div>
                    </div>
                </div>
            `;
            listWrapper.appendChild(card);
        });
        
        container.appendChild(listWrapper);
    } catch (err) {
        container.innerHTML = `<div class="chart-placeholder" style="color: var(--accent-red)">Error loading aggregates: ${err.message}</div>`;
    }
}

// MLOps Dashboard & Retraining Functions
let telemetryData = null;
let retrainPollingId = null;

async function fetchTelemetryStats() {
    try {
        const response = await fetch(`${state.apiBaseUrl}/data/export`);
        if (!response.ok) throw new Error('Failed to fetch telemetry dataset');
        telemetryData = await response.json();
        
        const count = telemetryData.length;
        elements.telemetryCountVal.textContent = count;
        
        if (count > 0) {
            elements.downloadDatasetBtn.disabled = false;
        } else {
            elements.downloadDatasetBtn.disabled = true;
        }
    } catch (err) {
        console.error("Error fetching telemetry dataset stats:", err);
        elements.telemetryCountVal.textContent = 'Error';
        elements.downloadDatasetBtn.disabled = true;
    }
}

async function triggerRetraining() {
    try {
        elements.triggerRetrainBtn.disabled = true;
        elements.triggerRetrainBtn.textContent = 'Starting Retraining...';
        
        const response = await fetch(`${state.apiBaseUrl}/train`, { method: 'POST' });
        const data = await response.json();
        
        if (data.status === 'success') {
            updateRetrainUI(true, 'Starting...');
            elements.retrainConsoleLog.textContent = 'Subprocess launched. Waiting for training logs...\n';
            startRetrainPolling();
        } else {
            alert(`Retraining failed to start: ${data.message}`);
            elements.triggerRetrainBtn.disabled = false;
            elements.triggerRetrainBtn.textContent = 'Run Model Retraining';
        }
    } catch (err) {
        console.error("Error triggering model retraining:", err);
        alert("Failed to connect to API retraining endpoint.");
        elements.triggerRetrainBtn.disabled = false;
        elements.triggerRetrainBtn.textContent = 'Run Model Retraining';
    }
}

function startRetrainPolling() {
    if (retrainPollingId) clearInterval(retrainPollingId);
    retrainPollingId = setInterval(pollRetrainingStatus, 2000);
}

async function pollRetrainingStatus() {
    try {
        const statusResponse = await fetch(`${state.apiBaseUrl}/train/status`);
        const statusData = await statusResponse.json();
        
        const logsResponse = await fetch(`${state.apiBaseUrl}/train/logs`);
        const logsData = await logsResponse.json();
        
        elements.retrainConsoleLog.textContent = logsData.logs;
        elements.retrainConsoleLog.scrollTop = elements.retrainConsoleLog.scrollHeight;
        
        if (!statusData.is_training) {
            clearInterval(retrainPollingId);
            retrainPollingId = null;
            
            updateRetrainUI(false, 'Idle');
            elements.retrainConsoleLog.textContent += '\n===================================\n[SUCCESS] Retraining complete. New weights loaded in API.\n';
            elements.retrainConsoleLog.scrollTop = elements.retrainConsoleLog.scrollHeight;
            
            checkServerHealth();
        } else {
            updateRetrainUI(true, 'Training...');
        }
    } catch (err) {
        console.error("Error polling retraining status:", err);
    }
}

function updateRetrainUI(isTraining, statusText) {
    elements.retrainStatusText.textContent = isTraining ? 'Retraining...' : 'Idle';
    elements.retrainStatusText.className = isTraining ? 'badge badge-training' : 'badge badge-idle';
    
    if (isTraining) {
        elements.retrainProgressWrapper.style.display = 'block';
        let pct = parseFloat(elements.retrainProgressBar.style.width) || 0;
        if (pct < 95) {
            pct += (95 - pct) * 0.05;
            elements.retrainProgressBar.style.width = `${pct}%`;
        }
        elements.triggerRetrainBtn.disabled = true;
        elements.triggerRetrainBtn.textContent = 'Retraining in Progress...';
    } else {
        elements.retrainProgressWrapper.style.display = 'none';
        elements.retrainProgressBar.style.width = '0%';
        elements.triggerRetrainBtn.disabled = false;
        elements.triggerRetrainBtn.textContent = 'Run Model Retraining';
    }
}

// Model Architecture Visualization Data
const modelArchitectures = {
    pointer: {
        title: "PointerNetwork (Project Implementation)",
        color: "var(--accent-cyan)",
        layers: [
            {
                id: "p_input",
                label: "Input Coords",
                shape: "rect",
                x: 100, y: 20, w: 200, h: 30,
                dim: "(Batch, N, 2)",
                title: "Input City Coordinates",
                desc: "Normalized 2D coordinates (x, y) representing Cartesian locations of N cities on the unit grid."
            },
            {
                id: "p_embed",
                label: "Linear Node Embed",
                shape: "rect",
                x: 100, y: 85, w: 200, h: 35,
                dim: "(Batch, N, 128)",
                title: "Linear Embedding Layer",
                desc: "nn.Linear(2, 128). Projects 2D coordinate inputs into a 128-dimensional continuous vector representation."
            },
            {
                id: "p_encoder",
                label: "Transformer Encoder",
                shape: "rect",
                x: 100, y: 155, w: 200, h: 45,
                dim: "(Batch, N, 128)",
                title: "Transformer Encoder (3 Layers)",
                desc: "nn.TransformerEncoder with 3 layers. Processes spatial embeddings using multi-head self-attention. No positional encodings are used to preserve permutation invariance of the input set."
            },
            {
                id: "p_query",
                label: "Decoder Query Maker",
                shape: "rect",
                x: 100, y: 235, w: 200, h: 40,
                dim: "(Batch, 1, 128)",
                title: "Decoder Query Projection",
                desc: "Concatenates the current city embedding (h_curr), the start city embedding (h_start), and the mean context (h_global). Projects (128*3) dimensions down to 128 dimensions."
            },
            {
                id: "p_pointer",
                label: "Pointer Attention",
                shape: "circle",
                x: 200, y: 315, r: 25,
                dim: "(Batch, N)",
                title: "Pointer Attention Mechanism",
                desc: "Computes dot-product attention scores between the decoder query q and the encoder's city representations h. The scores represent probability logits for visiting each city next."
            },
            {
                id: "p_mask",
                label: "Visited Masking & Logits",
                shape: "rect",
                x: 100, y: 375, w: 200, h: 35,
                dim: "(Batch, N)",
                title: "Visited Masking Loop",
                desc: "Applies mask to already visited nodes (setting scores to -1e9) to prevent duplicate visits. Passes output through LogSoftmax to yield valid probability distribution."
            }
        ],
        edges: [
            { from: "p_input", to: "p_embed" },
            { from: "p_embed", to: "p_encoder" },
            { from: "p_encoder", to: "p_query" },
            { from: "p_encoder", to: "p_pointer" },
            { from: "p_query", to: "p_pointer" },
            { from: "p_pointer", to: "p_mask" }
        ]
    },
    transformer: {
        title: "TSP_Transformer (Notebook Model)",
        color: "var(--accent-purple)",
        layers: [
            {
                id: "t_input_enc",
                label: "Input Coords",
                shape: "rect",
                x: 25, y: 20, w: 160, h: 30,
                dim: "(Batch, N, 2)",
                title: "Input Coordinates",
                desc: "2D city coordinate inputs (x, y) passed to the encoder."
            },
            {
                id: "t_input_dec",
                label: "Input Tour Indices",
                shape: "rect",
                x: 215, y: 20, w: 160, h: 30,
                dim: "(Batch, T)",
                title: "Decoder Input Indices",
                desc: "Sequence of indices representing the partially built tour (vocabulary tokens)."
            },
            {
                id: "t_embed_enc",
                label: "Cartesian Embedding",
                shape: "rect",
                x: 25, y: 85, w: 160, h: 35,
                dim: "(Batch, N, d_enc)",
                title: "Cartesian Coordinate Embedding",
                desc: "nn.Linear(2, d_model_enc). Projects raw 2D coordinates into encoder space."
            },
            {
                id: "t_embed_dec",
                label: "City Embedding & PE",
                shape: "rect",
                x: 215, y: 85, w: 160, h: 35,
                dim: "(Batch, T, d_dec)",
                title: "City Embedding + Positional Encoding",
                desc: "nn.Embedding(num_cities, d_model_dec) followed by sinusoidal PositionalEncoding. Adds sequential order context to the partial tour tokens."
            },
            {
                id: "t_encoder",
                label: "Transformer Encoder",
                shape: "rect",
                x: 25, y: 155, w: 160, h: 40,
                dim: "(Batch, N, d_enc)",
                title: "Transformer Encoder",
                desc: "nn.TransformerEncoder. Self-attention processing of spatial coordinate representations."
            },
            {
                id: "t_prj",
                label: "Dimension Projection",
                shape: "rect",
                x: 25, y: 225, w: 160, h: 35,
                dim: "(Batch, N, d_dec)",
                title: "Encoder-Decoder Projection",
                desc: "nn.Linear(d_model_enc, d_model_dec). Projects encoder output features to match decoder cross-attention size."
            },
            {
                id: "t_decoder",
                label: "Transformer Decoder",
                shape: "rect",
                x: 215, y: 290, w: 160, h: 45,
                dim: "(Batch, T, d_dec)",
                title: "Transformer Decoder",
                desc: "nn.TransformerDecoder. Decodes the partial tour sequence using self-attention (with causal mask to hide future steps) and cross-attention with encoder output."
            },
            {
                id: "t_output",
                label: "Linear Output Classifier",
                shape: "rect",
                x: 100, y: 375, w: 200, h: 40,
                dim: "(Batch, T, num_cities)",
                title: "Linear Classifier Layer",
                desc: "nn.Linear(d_model_dec, num_cities). Projects decoder output features directly to city logits. Flaw: Behaves like a language model and does not prevent duplicate visits."
            }
        ],
        edges: [
            { from: "t_input_enc", to: "t_embed_enc" },
            { from: "t_input_dec", to: "t_embed_dec" },
            { from: "t_embed_enc", to: "t_encoder" },
            { from: "t_encoder", to: "t_prj" },
            { from: "t_embed_dec", to: "t_decoder" },
            { from: "t_prj", to: "t_decoder" },
            { from: "t_decoder", to: "t_output" }
        ]
    }
};

let activeModelKey = "pointer";
let selectedLayerId = null;

function initArchitectureVisualizer() {
    // Add event listeners to toggle buttons
    document.querySelectorAll('.model-toggle-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.model-toggle-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const modelKey = btn.getAttribute('data-model');
            activeModelKey = modelKey;
            selectedLayerId = null;
            
            renderArchitectureDiagram();
            resetLayerInspector();
        });
    });
    
    renderArchitectureDiagram();
}

function resetLayerInspector() {
    const titleEl = document.getElementById('layer-title');
    const descEl = document.getElementById('layer-desc');
    
    titleEl.innerHTML = `Layer Inspector`;
    descEl.textContent = `Click on any layer/operation block in the diagram to inspect inputs, outputs, mathematical operations, and features.`;
}

function renderArchitectureDiagram() {
    const wrapper = document.getElementById('architecture-diagram-wrapper');
    if (!wrapper) return;
    
    const arch = modelArchitectures[activeModelKey];
    const width = 400;
    const height = 430;
    
    let svgHtml = `<svg class="arch-svg" viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">`;
    
    // Draw markers for arrow heads
    svgHtml += `
        <defs>
            <marker id="arrow-${activeModelKey}" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1.5 L 10 5 L 0 8.5 z" fill="${arch.color}" />
            </marker>
            <marker id="arrow-gray" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1.5 L 10 5 L 0 8.5 z" class="arch-arrow-gray" />
            </marker>
        </defs>
    `;
    
    // Helper to find layer by id
    const findLayer = (id) => arch.layers.find(l => l.id === id);
    
    // Draw edges (connections)
    arch.edges.forEach(edge => {
        const fromNode = findLayer(edge.from);
        const toNode = findLayer(edge.to);
        if (!fromNode || !toNode) return;
        
        let x1, y1, x2, y2;
        
        if (fromNode.shape === "circle") {
            x1 = fromNode.x;
            y1 = fromNode.y + fromNode.r;
        } else {
            x1 = fromNode.x + fromNode.w / 2;
            y1 = fromNode.y + fromNode.h;
        }
        
        if (toNode.shape === "circle") {
            x2 = toNode.x;
            y2 = toNode.y - toNode.r;
        } else {
            x2 = toNode.x + toNode.w / 2;
            y2 = toNode.y;
        }
        
        // Custom routing for query context projection to attention node
        if (fromNode.id === "p_encoder" && toNode.id === "p_pointer") {
            // Route around
            svgHtml += `<path d="M ${fromNode.x + fromNode.w} ${fromNode.y + fromNode.h / 2} C ${fromNode.x + fromNode.w + 60} ${fromNode.y + fromNode.h / 2}, ${toNode.x + 60} ${toNode.y}, ${toNode.x + toNode.r} ${toNode.y}" 
                fill="none" class="arch-edge-gray" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrow-gray)" />`;
            return;
        }
        
        if (fromNode.id === "t_prj" && toNode.id === "t_decoder") {
            svgHtml += `<path d="M ${x1} ${y1} C ${x1} ${y1 + 30}, ${x2 - 80} ${y2 - 20}, ${x2} ${y2}" 
                fill="none" class="arch-edge-gray" stroke-width="1.5" stroke-dasharray="3 3" marker-end="url(#arrow-gray)" />`;
            return;
        }
        
        svgHtml += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="arch-edge" stroke-width="1.5" marker-end="url(#arrow-${activeModelKey})" />`;
    });
    
    // Draw layers/nodes
    arch.layers.forEach(layer => {
        const isSelected = (layer.id === selectedLayerId);
        const nodeClass = `arch-node ${isSelected ? 'selected' : ''}`;
        const strokeColor = arch.color;
        const fillColor = isSelected ? "rgba(255, 255, 255, 0.08)" : "rgba(13, 16, 23, 0.9)";
        
        svgHtml += `<g class="${nodeClass}" id="${layer.id}" style="--glow-color: ${strokeColor}">`;
        
        if (layer.shape === "circle") {
            svgHtml += `<circle cx="${layer.x}" cy="${layer.y}" r="${layer.r}" fill="${fillColor}" stroke="${strokeColor}" stroke-width="1.5" />`;
            svgHtml += `<text x="${layer.x}" y="${layer.y - 2}" text-anchor="middle" dominant-baseline="middle" fill="#ffffff" font-size="9" font-family="var(--font-sans)" font-weight="600">${layer.label.split(' ')[0]}</text>`;
            svgHtml += `<text x="${layer.x}" y="${layer.y + 8}" text-anchor="middle" dominant-baseline="middle" fill="#ffffff" font-size="9" font-family="var(--font-sans)" font-weight="600">${layer.label.split(' ')[1] || ''}</text>`;
        } else {
            svgHtml += `<rect x="${layer.x}" y="${layer.y}" width="${layer.w}" height="${layer.h}" rx="6" fill="${fillColor}" stroke="${strokeColor}" stroke-width="1.5" />`;
            svgHtml += `<text x="${layer.x + layer.w / 2}" y="${layer.y + layer.h / 2}" text-anchor="middle" dominant-baseline="middle" fill="#ffffff" font-size="10" font-family="var(--font-sans)" font-weight="600">${layer.label}</text>`;
        }
        
        svgHtml += `</g>`;
    });
    
    svgHtml += `</svg>`;
    wrapper.innerHTML = svgHtml;
    
    // Register event listeners on svg nodes
    arch.layers.forEach(layer => {
        const el = document.getElementById(layer.id);
        if (el) {
            el.addEventListener('click', () => {
                selectedLayerId = layer.id;
                
                // Redraw diagram to show selection border
                renderArchitectureDiagram();
                
                // Update detail card
                const titleEl = document.getElementById('layer-title');
                const descEl = document.getElementById('layer-desc');
                
                titleEl.innerHTML = `
                    <span>${layer.title}</span>
                    <span class="dimension-badge">${layer.dim}</span>
                `;
                descEl.textContent = layer.desc;
            });
        }
    });
}
