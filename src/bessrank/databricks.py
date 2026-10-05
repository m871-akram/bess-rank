"""Minimal Databricks REST helpers (Unity Catalog, Files, Workspace, Jobs) and the S0 smoke test.

Authentication: in the Claude Code cloud session the proxy attaches the token for
DATABRICKS_HOST. If DATABRICKS_TOKEN is set it is sent instead. The token is never printed,
logged or written to a file (CLAUDE.md rule 11).
"""
import base64
import json
import os
import time

import requests

from bessrank import config

SMOKE_NOTEBOOK = config.ROOT / "notebooks" / "00_smoke_test.py"


def _host():
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    if not host:
        raise RuntimeError("DATABRICKS_HOST is not set")
    return host if host.startswith("http") else "https://" + host


def _headers():
    token = os.environ.get("DATABRICKS_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api(method, path, **kwargs):
    """Call the REST API and return the response; HTTP errors are left to the caller."""
    kwargs.setdefault("timeout", 120)
    headers = {**_headers(), **kwargs.pop("headers", {})}
    return requests.request(method, _host() + path, headers=headers, **kwargs)


def _check(resp, what):
    """Raise with the status and Databricks' error message if a call failed."""
    if resp.status_code != 200:
        raise RuntimeError(f"{what}: HTTP {resp.status_code} {resp.text[:300]}")
    return resp.json() if resp.content else {}


def available():
    """True if DATABRICKS_HOST is set and authentication works."""
    if not os.environ.get("DATABRICKS_HOST"):
        return False
    try:
        return api("GET", "/api/2.0/preview/scim/v2/Me", timeout=30).status_code == 200
    except requests.RequestException:
        return False


def current_user():
    return _check(api("GET", "/api/2.0/preview/scim/v2/Me"), "authentication")["userName"]


# --- Unity Catalog ------------------------------------------------------------------------
def ensure_schema():
    """Create workspace.bess if missing. Returns 'exists' or 'created'."""
    full_name = f"{config.DBX_CATALOG}.{config.DBX_SCHEMA}"
    if api("GET", f"/api/2.1/unity-catalog/schemas/{full_name}").status_code == 200:
        return "exists"
    body = {"name": config.DBX_SCHEMA, "catalog_name": config.DBX_CATALOG, "comment": "bess-rank project"}
    _check(api("POST", "/api/2.1/unity-catalog/schemas", json=body), "create schema")
    return "created"


def ensure_volume():
    """Create the managed volume workspace.bess.raw if missing. Returns 'exists' or 'created'."""
    full_name = f"{config.DBX_CATALOG}.{config.DBX_SCHEMA}.{config.DBX_VOLUME}"
    if api("GET", f"/api/2.1/unity-catalog/volumes/{full_name}").status_code == 200:
        return "exists"
    body = {
        "catalog_name": config.DBX_CATALOG,
        "schema_name": config.DBX_SCHEMA,
        "name": config.DBX_VOLUME,
        "volume_type": "MANAGED",
    }
    _check(api("POST", "/api/2.1/unity-catalog/volumes", json=body), "create volume")
    return "created"


# --- Files API (volume) -------------------------------------------------------------------
def upload_file(local_path, volume_relpath):
    """Upload a local file to the volume (overwrites). True on success."""
    remote = f"{config.DBX_VOLUME_PATH}/{volume_relpath}"
    folder = remote.rsplit("/", 1)[0]
    api("PUT", f"/api/2.0/fs/directories{folder}")  # idempotent
    with open(local_path, "rb") as f:
        resp = api("PUT", f"/api/2.0/fs/files{remote}", params={"overwrite": "true"}, data=f,
                   headers={"Content-Type": "application/octet-stream"})
    if resp.status_code not in (200, 204):
        print(f"Upload of {volume_relpath} failed: HTTP {resp.status_code} {resp.text[:200]}")
        return False
    return True


def download_file(volume_relpath, local_path):
    """Download a file from the volume. False if it is missing or the call fails."""
    resp = api("GET", f"/api/2.0/fs/files{config.DBX_VOLUME_PATH}/{volume_relpath}", stream=True)
    if resp.status_code != 200:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        for block in resp.iter_content(chunk_size=1 << 20):
            f.write(block)
    return True


# --- Workspace and Jobs -------------------------------------------------------------------
def import_notebook(local_path, workspace_path):
    """Import a Python source notebook into the workspace (overwrites)."""
    _check(api("POST", "/api/2.0/workspace/mkdirs", json={"path": workspace_path.rsplit("/", 1)[0]}), "mkdirs")
    body = {
        "path": workspace_path,
        "format": "SOURCE",
        "language": "PYTHON",
        "content": base64.b64encode(local_path.read_bytes()).decode(),
        "overwrite": True,
    }
    _check(api("POST", "/api/2.0/workspace/import", json=body), "import notebook")


def submit_notebook_run(workspace_path, run_name):
    """Submit a one-off notebook run on serverless compute (no cluster spec). Returns run_id."""
    body = {
        "run_name": run_name,
        "tasks": [{"task_key": "main", "notebook_task": {"notebook_path": workspace_path, "source": "WORKSPACE"}}],
    }
    return _check(api("POST", "/api/2.2/jobs/runs/submit", json=body), "submit run")["run_id"]


def wait_for_run(run_id, timeout_s=1800, poll_s=15):
    """Poll a run until it ends. Returns (result_state, notebook exit value or error)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        run = _check(api("GET", "/api/2.2/jobs/runs/get", params={"run_id": run_id}), "get run")
        state = run["state"]
        if state["life_cycle_state"] in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            task_run_id = run["tasks"][0]["run_id"]
            output = _check(api("GET", "/api/2.2/jobs/runs/get-output", params={"run_id": task_run_id}), "get output")
            detail = output.get("notebook_output", {}).get("result") or output.get("error") or state.get("state_message")
            return state.get("result_state", state["life_cycle_state"]), detail
        time.sleep(poll_s)
    return "TIMEOUT", f"run {run_id} still running after {timeout_s} s"


# --- S0 smoke test ------------------------------------------------------------------------
def smoke_test():
    """Check each step of the Databricks path and record pass/fail in results/databricks_smoke.json.

    Steps: authentication, schema, volume, parquet upload, serverless notebook run (reads the
    parquet, trains a small XGBoost on training-period days, logs to MLflow, writes a Delta
    table). Stops at the first failing step.
    """
    steps = []
    user = None

    def record(step, ok, detail=""):
        if user:  # keep the account e-mail out of the committed results file
            detail = detail.replace(user, "<user>")
        steps.append({"step": step, "ok": ok, "detail": detail})
        print(f"[{'ok' if ok else 'FAILED'}] {step} {detail}", flush=True)
        return ok

    try:
        user = current_user()
        record("authentication", True)  # the user name is not written to results/
        record("schema workspace.bess", True, ensure_schema())
        record("volume workspace.bess.raw", True, ensure_volume())
        if not config.HOURLY_PARQUET.exists():
            raise RuntimeError("data/processed/hourly.parquet is missing; run `python -m bessrank.run data` first")
        ok = upload_file(config.HOURLY_PARQUET, "processed/hourly.parquet") and upload_file(
            config.HOURLY_META, "processed/hourly_meta.json")
        if not record("upload processed parquet", ok):
            raise RuntimeError("upload failed")
        # Not /Users/<user>/bess-rank: that path is reserved for the MLflow experiment.
        workspace_path = f"/Users/{user}/bess-rank-notebooks/00_smoke_test"
        import_notebook(SMOKE_NOTEBOOK, workspace_path)
        record("import notebook", True)
        run_id = submit_notebook_run(workspace_path, "bess-rank S0 smoke test")
        record("submit serverless run", True, f"run_id {run_id}")
        started = time.time()
        result, detail = wait_for_run(run_id)
        record("serverless run", result == "SUCCESS", f"{result} after {time.time() - started:.0f} s: {detail}")
    except Exception as exc:  # report the failing step instead of a traceback
        record("error", False, str(exc)[:400])

    out = config.RESULTS_DIR / "databricks_smoke.json"
    out.write_text(json.dumps({"checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "steps": steps}, indent=2))
    return all(s["ok"] for s in steps)
