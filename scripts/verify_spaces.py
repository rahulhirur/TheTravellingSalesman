import os
import time
import urllib.request
import urllib.error
import json
import sys

def clean_space_id(raw_str):
    if not raw_str:
        return None
    val = raw_str.strip()
    if "spaces/" in val:
        val = val.split("spaces/")[-1]
    elif "huggingface.co/" in val:
        val = val.split("huggingface.co/")[-1]
    return val.strip("/")

def print_space_logs(space, token, build_type="run"):
    print(f"\n=== FETCHING LIVE LOGS FOR: {space} ({build_type.upper()}) ===")
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = f"https://huggingface.co/api/spaces/{space}/logs/{build_type}"
        req = urllib.request.Request(url, headers=headers)
        
        # Read SSE stream line-by-line to prevent connection hanging
        with urllib.request.urlopen(req, timeout=10) as resp:
            lines = []
            for line in resp:
                decoded = line.decode(errors="ignore").strip()
                if decoded.startswith("data:"):
                    try:
                        # Extract SSE event content JSON
                        data_json = json.loads(decoded[5:].strip())
                        lines.append(data_json.get("data", ""))
                    except Exception:
                        lines.append(decoded)
                elif decoded:
                    lines.append(decoded)
                
                # Retrieve last 150 lines then close to avoid hanging
                if len(lines) >= 150:
                    break
                    
        # Print clean log lines
        for line in lines[-100:]:
            print(line, end="")
    except Exception as e:
        print(f"Failed to retrieve Space logs: {e}")
    print("=========================================\n")

def test_inference(base_url, token):
    print(f"  Running Layer 3: End-to-End Inference Test...")
    solve_url = f"{base_url}/solve"
    
    # 3-point dummy TSP problem matching SolveRequest schema
    payload = {
        "points": [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        "solvers": ["nearest_neighbor"]
    }
    
    try:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
            
        req = urllib.request.Request(
            solve_url,
            data=data,
            headers=headers,
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=20) as resp:
            resp_data = json.loads(resp.read().decode())
            
        results = resp_data.get("results", {})
        nn_result = results.get("nearest_neighbor")
        
        if nn_result and "path" in nn_result and "distance" in nn_result:
            print(f"  [SUCCESS] Inference Test passed! Solution found: {nn_result}")
            return True
        else:
            print(f"  [FAILURE] Inference Test returned invalid schema: {resp_data}")
            return False
            
    except Exception as e:
        print(f"  [FAILURE] Inference Test crashed: {e}")
        return False

def check_spaces():
    hf_token = os.environ.get("HF_TOKEN")
    spaces = []
    
    frontend = clean_space_id(os.environ.get("HF_SPACE_FRONTEND"))
    if frontend:
        spaces.append(frontend)
        
    backend = clean_space_id(os.environ.get("HF_SPACE_BACKEND"))
    if backend:
        spaces.append(backend)
        
    if not spaces:
        print("No Hugging Face Spaces defined to check.")
        sys.exit(0)
        
    max_attempts = 35
    check_interval = 15
    print(f"Monitoring Hugging Face Spaces: {spaces}")
    
    for space in spaces:
        print(f"\nChecking status for Space: {space}...")
        req_url = f"https://huggingface.co/api/spaces/{space}"
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        req = urllib.request.Request(req_url, headers=headers)
        
        success = False
        for attempt in range(1, max_attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                
                runtime = data.get("runtime", {})
                stage = runtime.get("stage", "UNKNOWN")
                sdk = runtime.get("sdk", "unknown-sdk")
                hardware = runtime.get("hardware", "unknown-hardware")
                
                print(f"  Attempt {attempt}/{max_attempts} | Stage: {stage} | SDK: {sdk} | Hardware: {hardware}")
                
                info_msg = runtime.get("error_message")
                if info_msg:
                    print(f"  [HF Info/Error] {info_msg}")
                
                if stage == "RUNNING":
                    print(f"  [SUCCESS] Space {space} is running on Hugging Face!")
                    
                    # Layer 2: HTTP Health Check
                    parts = space.split("/")
                    if len(parts) == 2:
                        username = parts[0].lower().replace("_", "-")
                        space_name = parts[1].lower().replace("_", "-")
                        base_url = f"https://{username}-{space_name}.hf.space"
                        
                        health_url = base_url
                        is_backend = "api" in space_name
                        if is_backend:
                            health_url += "/health"
                        print(f"  Running Layer 2: HTTP Health check at: {health_url}...")
                        try:
                            health_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                            if hf_token:
                                health_headers["Authorization"] = f"Bearer {hf_token}"
                                
                            http_req = urllib.request.Request(
                                health_url, 
                                headers=health_headers
                            )
                            with urllib.request.urlopen(http_req, timeout=15) as http_resp:
                                code = http_resp.getcode()
                                print(f"  [SUCCESS] HTTP Health Check responded with: {code}")
                        except urllib.error.HTTPError as http_err:
                            if is_backend:
                                print(f"  [FAILURE] Backend Health Check failed: {http_err.code} - {http_err.reason}")
                                sys.exit(1)
                            else:
                                print(f"  [SUCCESS] Frontend Health Check responded with code: {http_err.code} (Server is live!)")
                        except Exception as http_err:
                            print(f"  [WARNING] HTTP Health Check failed: {http_err}. Retrying...")
                            time.sleep(check_interval)
                            continue
                            
                        # Layer 3: Inference Test (Backend Only)
                        if is_backend:
                            if not test_inference(base_url, hf_token):
                                sys.exit(1)
                                
                        success = True
                        break
                    else:
                        success = True
                        break
                        
                elif stage in ["BUILD_ERROR", "RUNTIME_ERROR"]:
                    print(f"\n[FAILURE] Space {space} failed with stage: {stage}!")
                    log_type = "build" if stage == "BUILD_ERROR" else "run"
                    print_space_logs(space, hf_token, log_type)
                    sys.exit(1)
                elif stage == "PAUSED":
                    print(f"\n[WARNING] Space {space} is paused.")
                    print_space_logs(space, hf_token, "run")
                    sys.exit(1)
                    
            except urllib.error.HTTPError as e:
                print(f"  Attempt {attempt}/{max_attempts} | HTTP Error: {e.code} - {e.reason}")
                if e.code == 404:
                    print(f"[ERROR] Space {space} not found! Check space ID and permissions.")
                    sys.exit(1)
            except Exception as e:
                print(f"  Attempt {attempt}/{max_attempts} | Request failed: {e}")
            
            time.sleep(check_interval)
            
        if not success:
            print(f"\n[TIMEOUT] Space {space} did not reach RUNNING stage in time.")
            print_space_logs(space, hf_token, "build")
            print_space_logs(space, hf_token, "run")
            sys.exit(1)
            
    print("\nAll Hugging Face Spaces deployed and running successfully!")

if __name__ == "__main__":
    check_spaces()
