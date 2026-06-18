"""
GEE Web App – FastAPI backend

"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import calendar
import io
import json
import logging
import os
import re
import secrets
import signal
import string
import subprocess
import threading
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Optional

import duckdb
import geopandas as gpd
import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from workflow.products import PRODUCT_REGISTRY
from workflow.time_chunks import get_time_chunks


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _force_unlink(path: Path, retries: int = 5, delay: float = 1.0) -> bool:
    """Delete a file, retrying on PermissionError for transient Windows locks.
    Returns True if deleted, False if all attempts failed."""
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
    return False


# ─── Paths ────────────────────────────────────────────────────────────────────

_docker_data = Path("/app/data")
BASE_DATA_DIR = _docker_data if _docker_data.exists() else Path(__file__).parent.parent / "data"
RUNS_DIR      = BASE_DATA_DIR / "runs"
CONFIG_DIR    = Path(tempfile.gettempdir()) / "gee_configs"
GEE_KEY_PATH  = Path(
    os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        str(Path(__file__).parent.parent / "config" / "gee-key.json"),
    )
).resolve()
APP_DIR          = Path(__file__).parent.parent        # repo root
SNAKEFILE        = APP_DIR / "Snakefile"
LOG_HANDLER      = APP_DIR / "scripts" / "snakemake_log_handler.py"
RUN_DB_PATH      = RUNS_DIR / "run_state.duckdb"
SNAKEMAKE_PIDFILE = APP_DIR / ".snakemake.pid"  # written on every launch for stop-app cleanup

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

for d in [RUNS_DIR, CONFIG_DIR, Path.home() / ".duckdb"]:
    d.mkdir(parents=True, exist_ok=True)

_REQUIRED_KEY_FIELDS = {"type", "project_id", "private_key", "client_email", "token_uri"}


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="GEE Web App API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic models ──────────────────────────────────────────────────────────

class ProductConfig(BaseModel):
    product: str
    bands: list[str]
    stats: list[str]
    date_start: str   # YYYY-MM-DD
    date_end: str     # YYYY-MM-DD

DEFAULT_GEE_CONCURRENCY = 2

class SubmitRunRequest(BaseModel):
    run_id: str
    products: list[ProductConfig]
    gee_concurrency: int = DEFAULT_GEE_CONCURRENCY
    id_column: str | None = None

class RetryRunRequest(BaseModel):
    gee_concurrency: int | None = None  # None → keep stored value

class ResumeRunRequest(BaseModel):
    gee_concurrency: int | None = None  # None → SIGCONT with no change


# ─── DuckDB helpers  ──────────────────────────────────

def _duckdb_connect(retries: int = 8, delay: float = 0.25) -> duckdb.DuckDBPyConnection:
    last_exc = None
    for attempt in range(retries):
        try:
            return duckdb.connect(str(RUN_DB_PATH))
        except duckdb.IOException as e:
            if "lock" not in str(e).lower():
                raise
            last_exc = e
            time.sleep(delay * (attempt + 1))
    raise last_exc  # type: ignore[misc]

def ensure_run_db():
    with _duckdb_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_status (
                run_id                VARCHAR PRIMARY KEY,
                status                VARCHAR,
                attempts              INTEGER,
                config_hash           VARCHAR,
                created_at            TIMESTAMP,
                updated_at            TIMESTAMP,
                last_error            VARCHAR,
                snakemake_pid         BIGINT,
                snakemake_log_path    VARCHAR,
                snakemake_config_path VARCHAR,
                snakefile             VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_events (
                event_time   TIMESTAMP,
                run_id       VARCHAR,
                event_type   VARCHAR,
                status       VARCHAR,
                message      VARCHAR,
                payload_json VARCHAR
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                run_id      VARCHAR,
                product     VARCHAR,
                band        VARCHAR,
                time_chunk  VARCHAR,
                status      VARCHAR DEFAULT 'pending',
                jobid       INTEGER,
                log_path    VARCHAR,
                started_at  TIMESTAMP,
                finished_at TIMESTAMP,
                error       VARCHAR,
                PRIMARY KEY (run_id, product, band, time_chunk)
            )
        """)

ensure_run_db()

def _upsert_run_status(run_id: str, record: dict):
    with _duckdb_connect() as conn:
        conn.execute("""
            INSERT INTO run_status (
                run_id, status, attempts, config_hash,
                created_at, updated_at, last_error,
                snakemake_pid, snakemake_log_path, snakemake_config_path, snakefile
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                status                = excluded.status,
                attempts              = excluded.attempts,
                config_hash           = excluded.config_hash,
                updated_at            = excluded.updated_at,
                last_error            = excluded.last_error,
                snakemake_pid         = excluded.snakemake_pid,
                snakemake_log_path    = excluded.snakemake_log_path,
                snakemake_config_path = excluded.snakemake_config_path,
                snakefile             = excluded.snakefile
        """, [
            run_id,
            record.get("status"),
            int(record.get("attempts", 0)),
            record.get("config_hash"),
            record.get("created_at"),
            record.get("updated_at"),
            record.get("last_error"),
            record.get("snakemake_pid"),
            record.get("snakemake_log_path"),
            record.get("snakemake_config_path"),
            record.get("snakefile", str(SNAKEFILE)),
        ])

def _append_event(run_id: str, event_type: str, status: str, message: str = "", payload=None):
    now = datetime.now(timezone.utc).isoformat()
    with _duckdb_connect() as conn:
        conn.execute("""
            INSERT INTO run_events (event_time, run_id, event_type, status, message, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [now, run_id, event_type, status, message, json.dumps(payload or {}, default=str)])

def _get_job_counts(run_id: str, meta: dict | None = None) -> dict:
    # Derive total/done from the filesystem — no DuckDB locking needed and always accurate.
    # A parquet chunk file existing means that band×chunk pair fully completed.
    if meta is None:
        meta = _load_yaml(run_id)
    payload  = (meta or {}).get("payload") or {}
    products = payload.get("products") or {}
    chunks_dir  = RUNS_DIR / run_id / "intermediate" / "chunks"
    results_dir = RUNS_DIR / run_id / "results"

    total = done = 0
    by_product: dict = {}
    for prod, cfg in products.items():
        # When a product finishes, Snakemake merges all chunk files into a final
        # parquet and removes the intermediates. Count all chunks for that product
        # as done based on the merged result so the progress bar doesn't reset.
        prod_merged = (results_dir / prod).exists() and any((results_dir / prod).glob("*.parquet"))
        prod_total = prod_done = 0
        for band in cfg.get("bands", []):
            band_merged = (chunks_dir / prod / f"merged_{band}.parquet").exists()
            for chunk in cfg.get("time_chunks", []):
                prod_total += 1
                if prod_merged or band_merged or (chunks_dir / prod / f"{band}_{chunk}.parquet").exists():
                    prod_done += 1
        total += prod_total
        done  += prod_done
        by_product[prod] = {"total": prod_total, "done": prod_done}

    # running/failed come from DuckDB, but are only trustworthy while Snakemake is
    # actively running.  When the process is gone the log handler can't update them,
    # so we zero them out and let the filesystem-based done/total drive the display.
    raw_status = meta.get("status", "unknown") if meta else "unknown"
    is_active = raw_status in ("running", "paused") and _is_pid_alive(meta.get("snakemake_pid") if meta else None)

    try:
        with _duckdb_connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) FROM jobs WHERE run_id=? AND status IN ('running','failed') GROUP BY status",
                [run_id],
            ).fetchall()
            shelved = int(conn.execute(
                "SELECT COUNT(*) FROM run_events WHERE run_id=? AND event_type='job_shelved'",
                [run_id],
            ).fetchone()[0])
        db_counts = {r[0]: int(r[1]) for r in rows}
    except Exception:
        db_counts = {}
        shelved = 0

    running = db_counts.get("running", 0) if is_active else 0
    failed  = db_counts.get("failed",  0) if is_active else 0
    pending = max(0, total - done - running - failed)
    return {"total": total, "done": done, "failed": failed, "running": running, "pending": pending, "shelved": shelved, "by_product": by_product}

def _initialise_jobs(run_id: str, payload: dict):
    """Pre-populate the jobs table (mirrors main.py initialise_jobs)."""
    products = payload.get("products", {}) or {}
    rows = []
    for prod, cfg in products.items():
        for band in cfg.get("bands", []):
            for chunk in cfg.get("time_chunks", []):
                chunk_path = (
                    RUNS_DIR / run_id / "intermediate" / "chunks" / prod / f"{band}_{chunk}.parquet"
                )
                status = "done" if chunk_path.exists() else "pending"
                rows.append((run_id, prod, band, chunk, status))
    if not rows:
        return
    with _duckdb_connect() as conn:
        conn.executemany("""
            INSERT INTO jobs (run_id, product, band, time_chunk, status) VALUES (?,?,?,?,?)
            ON CONFLICT (run_id, product, band, time_chunk) DO UPDATE SET
                status = CASE
                    WHEN excluded.status = 'done' THEN 'done'
                    WHEN jobs.status = 'done'     THEN 'done'
                    ELSE excluded.status
                END
        """, rows)

# ─── YAML helpers (mirrors main.py update_run_registry / load_run_registry) ───

def _run_yaml_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "run.yaml"

def _load_yaml(run_id: str) -> dict | None:
    p = _run_yaml_path(run_id)
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text()) or None

def _save_yaml(run_id: str, record: dict):
    p = _run_yaml_path(run_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(record, sort_keys=False))

def _update_registry(run_id: str, payload: dict, status: str,
                     config_hash: str | None = None,
                     error_message: str | None = None,
                     bump_attempt: bool = False,
                     clear_pid: bool = False):
    now      = datetime.now(timezone.utc).isoformat()
    existing = _load_yaml(run_id) or {}
    attempts = existing.get("attempts", 0)
    if bump_attempt:
        attempts += 1

    record = {
        "run_id":               run_id,
        "status":               status,
        "created_at":           existing.get("created_at", now),
        "updated_at":           now,
        "attempts":             attempts,
        "config_hash":          config_hash,
        "payload":              payload,
        "last_error":           error_message,
        # Clear the old PID when transitioning to running so that _resolve_status
        # doesn't see the dead PID from a previous attempt and immediately flip
        # the status back to failed before _set_execution_meta writes the new PID.
        # Also cleared on Windows pause (clear_pid=True) because the process is
        # killed outright — _resolve_status uses pid=None to distinguish an
        # intentional Windows pause from an unexpected crash.
        "snakemake_pid":        None if (status == "running" or clear_pid) else existing.get("snakemake_pid"),
        "snakemake_log_path":   existing.get("snakemake_log_path"),
        "snakemake_config_path":existing.get("snakemake_config_path"),
        "snakefile":            existing.get("snakefile", str(SNAKEFILE)),
    }
    if status == "running":
        record["last_started_at"] = now
    if status in {"completed", "failed"}:
        record["last_finished_at"] = now

    _save_yaml(run_id, record)
    _upsert_run_status(run_id, record)

    messages = {
        "queued":    "Run queued",
        "running":   f"Run started (attempt {attempts})",
        "completed": "Run completed successfully",
        "failed":    f"Run failed{': ' + error_message if error_message else ''}",
        "stopped":   "Run stopped by user",
        "paused":    "Run paused by user",
        "resumed":   "Run resumed by user",
    }
    _append_event(run_id, "status_change", status, messages.get(status, status))

def _set_execution_meta(run_id: str, pid: int, log_path: str, config_path: str):
    existing = _load_yaml(run_id) or {}
    existing["snakemake_pid"]          = pid
    existing["snakemake_log_path"]     = log_path
    existing["snakemake_config_path"]  = config_path
    existing["updated_at"]             = datetime.now(timezone.utc).isoformat()
    _save_yaml(run_id, existing)
    _upsert_run_status(run_id, existing)
    _append_event(run_id, "pipeline_started", existing.get("status", "running"),
                  f"Snakemake launched (PID {pid})", {"log_path": log_path, "config_path": config_path})

def _list_saved_runs() -> list[dict]:
    runs = []
    seen: set[str] = set()
    for p in sorted(RUNS_DIR.glob("*/run.yaml"), key=lambda x: x.stat().st_mtime, reverse=True):
        m = _load_yaml(p.parent.name)
        if m:
            runs.append(m)
            seen.add(p.parent.name)

    # Include runs that exist only in the DuckDB (no run.yaml on disk)
    try:
        with _duckdb_connect() as conn:
            rows = conn.execute(
                "SELECT run_id, status, created_at, updated_at FROM run_status ORDER BY created_at DESC"
            ).fetchall()
        for run_id, status, created_at, updated_at in rows:
            if run_id not in seen:
                # If the run directory no longer exists on disk, the user manually
                # deleted it — purge from DuckDB so the UI stops showing it.
                if not (RUNS_DIR / run_id).exists():
                    try:
                        with _duckdb_connect() as conn:
                            conn.execute("DELETE FROM run_status WHERE run_id = ?", [run_id])
                            conn.execute("DELETE FROM run_events WHERE run_id = ?", [run_id])
                            conn.execute("DELETE FROM jobs WHERE run_id = ?", [run_id])
                    except Exception:
                        pass
                    continue
                runs.append({
                    "run_id":     run_id,
                    "status":     status,
                    "created_at": _ts_utc(created_at) if created_at else "",
                    "updated_at": _ts_utc(updated_at) if updated_at else "",
                    "payload":    {},
                })
    except Exception:
        pass

    return runs

# ─── Process helpers ──────────────────────────────────────────────────────────

# SIGKILL doesn't exist on Windows; _signal_process_tree uses taskkill /F instead
# regardless of which termination signal is passed, so this constant is safe on both.
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)

def _is_pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except Exception:
        return False
    # Confirm the PID still belongs to a python/snakemake process, not a reused PID.
    try:
        if sys.platform == "win32":
            out = subprocess.check_output(
                ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            ).lower()
            return "python" in out
        else:
            cmdline = Path(f"/proc/{int(pid)}/cmdline").read_text().replace("\x00", " ").lower()
            return "snakemake" in cmdline or "python" in cmdline
    except Exception:
        return False

def _get_descendants(root_pid: int) -> list[int]:
    if sys.platform == "win32":
        try:
            # wmic returns columns alphabetically: ParentProcessId, ProcessId
            out = subprocess.check_output(
                ["wmic", "process", "get", "ParentProcessId,ProcessId"],
                stderr=subprocess.DEVNULL, text=True,
            )
            parent_map: dict[int, list[int]] = {}
            for line in out.splitlines()[1:]:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        ppid, pid = int(parts[0]), int(parts[1])
                        parent_map.setdefault(ppid, []).append(pid)
                    except ValueError:
                        continue
        except Exception:
            return []
    else:
        parent_map = {}
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                content = Path(f"/proc/{entry}/stat").read_text()
                after   = content.rsplit(")", 1)[1].strip().split()
                if len(after) >= 3:
                    parent_map.setdefault(int(after[1]), []).append(int(entry))
            except Exception:
                continue
    descendants, stack = [], [root_pid]
    while stack:
        cur = stack.pop()
        ch  = parent_map.get(cur, [])
        descendants.extend(ch)
        stack.extend(ch)
    return descendants

def _signal_process_tree(root_pid: int, sig: int):
    """Send signal to root PID and all its descendants."""
    if sys.platform == "win32":
        # Windows doesn't support SIGSTOP/SIGCONT/SIGKILL — use taskkill for termination.
        # Pause/resume are not supported on Windows; only termination is handled.
        _TERM_SIGS = {getattr(signal, "SIGTERM", 15), getattr(signal, "SIGKILL", 9)}
        if sig in _TERM_SIGS:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(root_pid)],
                    stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                )
            except Exception:
                pass
        return
    for child in _get_descendants(root_pid):
        try:
            os.kill(child, sig)
        except Exception:
            pass
    try:
        os.kill(root_pid, sig)
    except Exception:
        pass


def _filter_snakemake_output(proc: subprocess.Popen, log_path: Path):
    """Copy Snakemake output to log_path, suppressing the detail block for
    merge_product_parquet (input/output/jobid/reason/etc.) while keeping the
    rule header line itself."""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            skip_block = False
            for line in proc.stdout:
                if line.rstrip() == "rule merge_product_parquet:":
                    f.write(line)
                    f.flush()
                    skip_block = True
                    continue
                if skip_block:
                    if line.strip() == "":
                        skip_block = False
                    continue
                f.write(line)
                f.flush()
    except Exception:
        pass


def _launch_snakemake(run_id: str, payload: dict, log_path: Path) -> subprocess.Popen:
    """Start Snakemake for run_id. gee_concurrency is read from payload."""
    import base64, shutil
    run_dir     = RUNS_DIR / run_id
    cfg_path    = CONFIG_DIR / f"config_{uuid.uuid4().hex[:8]}.yaml"
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    # Determine which products are fully done (final parquet on disk).
    results_dir = run_dir / "results"
    done_products: set[str] = set()
    if results_dir.exists():
        for prod_dir in results_dir.iterdir():
            if prod_dir.is_dir() and any(prod_dir.glob("*.parquet")):
                done_products.add(prod_dir.name)

    # Clear incomplete markers. For non-done products also delete the partial
    # intermediate files (they may be corrupt from a killed job).
    incomplete_dir = run_dir / ".snakemake" / "incomplete"
    if incomplete_dir.exists():
        for marker in incomplete_dir.iterdir():
            try:
                partial_path = Path(base64.b64decode(marker.name).decode("utf-8"))
                if not any(prod in str(partial_path) for prod in done_products):
                    if not _force_unlink(partial_path):
                        continue  # lock held after all retries — leave marker for Snakemake
                marker.unlink(missing_ok=True)
            except Exception:
                pass  # leave marker intact — Snakemake will re-run this output on next resume
        try:
            incomplete_dir.rmdir()  # only succeeds if all markers were cleared
        except OSError:
            pass

    # Fix stale intermediate files for done products that would otherwise trigger
    # spurious mtime cascades.  Two cases arise when a run is killed mid-flight:
    #
    #  1. A temp() GeoJSON that extract_geojson_chunk was writing still exists on
    #     disk with a fresh mtime.  Because convert_to_parquet completed (metadata
    #     incomplete=False), the GeoJSON is no longer needed — delete it so it
    #     cannot trigger convert again.
    #
    #  2. A chunk parquet was re-produced by a cascade run *after* merge_band_chunks
    #     already finished (metadata endtime newer than the merged-band endtime).
    #     Delete it and back-date its metadata endtime so merge_band_chunks is not
    #     triggered again.
    if done_products:
        meta_root    = run_dir / ".snakemake" / "metadata"
        geojson_root = run_dir / "intermediate" / "geojson"
        chunk_root   = run_dir / "intermediate" / "chunks"

        if meta_root.exists():
            _meta_by_path: dict[str, Path] = {}
            for _mf in meta_root.iterdir():
                try:
                    _p = base64.b64decode(_mf.name).decode("utf-8")
                    _meta_by_path[_p.lower().replace("\\", "/")] = _mf
                except Exception:
                    pass

            def _read_meta(p: Path) -> dict:
                mf = _meta_by_path.get(str(p).lower().replace("\\", "/"))
                if not mf:
                    return {}
                try:
                    with open(mf) as _fh:
                        return json.load(_fh)
                except Exception:
                    return {}

            def _write_meta(p: Path, m: dict) -> None:
                mf = _meta_by_path.get(str(p).lower().replace("\\", "/"))
                if mf:
                    try:
                        with open(mf, "w") as _fh:
                            json.dump(m, _fh)
                    except Exception:
                        pass

            _chunk_re = re.compile(
                r"^(?P<band>.+?)_(?P<chunk>\d{4}-\d{2}(?:_\d{4}-\d{2})?|\d{4})\.parquet$"
            )

            for _prod in done_products:
                _ck_dir = chunk_root / _prod if chunk_root.exists() else None
                _gj_dir = geojson_root / _prod if geojson_root.exists() else None

                # Case 1: delete stale GeoJSON files whose chunk parquet is done.
                if _gj_dir and _gj_dir.is_dir() and _ck_dir:
                    for _gj in list(_gj_dir.glob("*.geojson")):
                        _ck_path = _ck_dir / (_gj.stem + ".parquet")
                        if _read_meta(_ck_path).get("incomplete") is False:
                            _gj.unlink(missing_ok=True)

                # Case 2: chunk parquet on disk with mtime newer than merged band.
                if _ck_dir and _ck_dir.is_dir():
                    for _ck in list(_ck_dir.glob("*.parquet")):
                        if _ck.name.startswith("merged_"):
                            continue
                        _m = _chunk_re.match(_ck.name)
                        if not _m:
                            continue
                        _merged_path = _ck_dir / f"merged_{_m.group('band')}.parquet"
                        _merged_meta = _read_meta(_merged_path)
                        if _merged_meta.get("incomplete") is not False:
                            continue
                        _merged_end = _merged_meta.get("endtime", 0)
                        if _merged_end and _ck.stat().st_mtime > _merged_end:
                            _ck.unlink(missing_ok=True)
                            _ck_meta = _read_meta(_ck)
                            if _ck_meta:
                                _ck_meta["endtime"] = _merged_end - 1
                                _write_meta(_ck, _ck_meta)

    try:
        extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
        subprocess.run(
            ["python", "-m", "snakemake", "--unlock", "--snakefile", str(SNAKEFILE),
             "--directory", str(run_dir)],
            capture_output=True, text=True, timeout=30, cwd=str(run_dir),
            **extra,
        )
    except Exception:
        pass

    gee_concurrency = int(payload.get("gee_concurrency") or DEFAULT_GEE_CONCURRENCY)
    env = {
        **os.environ,
        "GOOGLE_APPLICATION_CREDENTIALS": str(GEE_KEY_PATH),
        "GEE_RUN_ID":  run_id,
        "GEE_DB_PATH": str(RUN_DB_PATH),
        "HOME": str(Path.home()) if sys.platform == "win32" else tempfile.gettempdir(),
        "PYTHONWARNINGS": "ignore",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    cmd = [
        "python", "-W", "ignore::FutureWarning",
        "-m", "snakemake",
        "--snakefile",           str(SNAKEFILE),
        "--configfile",          str(cfg_path),
        "--directory",           str(run_dir),
        "-j",                    "12",
        "--resources",           f"gee={gee_concurrency}",
        "--rerun-triggers", "mtime",            # set to only mtime so when config is updated from GEE concurrency budget, finished products dont get rerun
        "--quiet",               "rules",   # we already have a logging script and snakemake default log is very messy
        "--keep-going",
        "--log-handler-script",  str(LOG_HANDLER),
    ]
    extra = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        close_fds=True, env=env, cwd=str(run_dir),
        **extra,
    )
    threading.Thread(
        target=_filter_snakemake_output, args=(proc, log_path), daemon=True
    ).start()
    _set_execution_meta(run_id, proc.pid, str(log_path), str(cfg_path))
    try:
        SNAKEMAKE_PIDFILE.write_text(str(proc.pid))
    except Exception:
        pass
    return proc


def _all_results_present(run_id: str | None, meta: dict | None) -> bool:
    """Return True if every product has its final merged parquet on disk."""
    if not run_id or not meta:
        return False
    payload  = (meta or {}).get("payload") or {}
    products = payload.get("products") or {}
    if not products:
        return False
    results_dir = RUNS_DIR / run_id / "results"
    for prod, cfg in products.items():
        start = cfg.get("start_date", "")
        end   = cfg.get("end_date", "")
        expected = results_dir / prod / f"{prod}_{start}_to_{end}.parquet"
        if not expected.exists():
            return False
    return True


def _resolve_status(meta: dict | None) -> str:
    if meta is None:
        return "unknown"
    status = meta.get("status", "unknown")
    if status == "paused":
        pid = meta.get("snakemake_pid")
        if pid is None:
            # Windows pause intentionally kills the process and clears the PID.
            # A None PID here means the pause was deliberate, not a crash.
            return "paused"
        if not _is_pid_alive(pid):
            run_id = meta.get("run_id")
            if run_id:
                _update_registry(run_id, meta.get("payload") or {}, status="failed",
                                 error_message="Process exited while paused")
            return "failed"
        return "paused"
    if status == "running":
        pid = meta.get("snakemake_pid")
        if pid is None:
            # PID not yet written — could still be in-flight, but if it's been
            # more than 60 s since launch the process almost certainly died before
            # it could write its PID (e.g. server restart).
            started = meta.get("last_started_at")
            if started:
                try:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
                    if age < 60:
                        return "running"
                except Exception:
                    pass
            # No timestamp or too old — treat as failed so it doesn't block new submissions.
            run_id = meta.get("run_id")
            if run_id:
                _update_registry(run_id, meta.get("payload") or {}, status="failed",
                                 error_message="Run process exited before PID was recorded")
            return "failed"
        if not _is_pid_alive(pid):
            # Process is gone — infer final status.
            # pid was set so there's no retry race condition; safe to persist.
            run_id = meta.get("run_id")
            # Check final result files first — Snakemake removes intermediate chunk
            # files on success, so _get_job_counts().done will be 0 for a completed run.
            resolved = "completed" if _all_results_present(run_id, meta) else None
            if resolved is None:
                counts = _get_job_counts(run_id, meta) if run_id else {}
                total  = counts.get("total", 0)
                done   = counts.get("done", 0)
                resolved = "completed" if (total > 0 and done == total) else "failed"
            if run_id:
                error_msg = None if resolved == "completed" else "Pipeline process exited unexpectedly"
                _update_registry(run_id, meta.get("payload") or {}, status=resolved,
                                 error_message=error_msg)
            return resolved
    return status

# ─── GEE key validation ───────────────────────────────────────────────────────

def _validate_gee_key(data: dict) -> str:
    missing = _REQUIRED_KEY_FIELDS - data.keys()
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(sorted(missing))}")
    if data.get("type") != "service_account":
        raise ValueError(f"Expected type 'service_account', got '{data.get('type')}'")
    email = data.get("client_email", "")
    if not email or "@" not in email:
        raise ValueError("client_email is empty or malformed")
    return email

# ─── Run result file helpers ──────────────────────────────────────────────────

def _results_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id / "results"

def _fix_payload_paths(run_id: str, payload: dict) -> dict:
    """
    Rewrite stored absolute paths that may be stale after the project folder
    is moved.  app_dir and output_dir are always derived from current globals;
    shp_path falls back to the run's inputs directory when the stored path
    no longer exists.
    """
    payload["app_dir"]    = APP_DIR.as_posix()
    payload["output_dir"] = _results_dir(run_id).as_posix()
    shp = payload.get("shp_path")
    if isinstance(shp, str):
        shp = shp.replace("\\", "/")  # normalise Windows backslashes before path operations
        p = Path(shp)
        if not p.exists():
            p = RUNS_DIR / run_id / "inputs" / p.name
        payload["shp_path"] = p.as_posix()
    return payload

def _list_result_products(run_id: str) -> list[str]:
    rdir = _results_dir(run_id)
    if not rdir.exists():
        return []
    return [p.name for p in rdir.iterdir() if p.is_dir() and p.name != "partial_checkout"]

def _list_finished_products(run_id: str) -> list[str]:
    """Products that have a merged parquet in results/<prod>/ (i.e. merge is complete)."""
    rdir = _results_dir(run_id)
    if not rdir.exists():
        return []
    return [
        p.name for p in rdir.iterdir()
        if p.is_dir() and p.name != "partial_checkout" and any(p.glob("*.parquet"))
    ]

def _find_product_parquet(run_id: str, product: str) -> Path:
    rdir  = _results_dir(run_id) / product
    files = sorted(rdir.glob("*.parquet")) if rdir.exists() else []
    if not files:
        raise HTTPException(404, f"No parquet for product '{product}' in run '{run_id}'")
    return files[0]

def _run_to_summary(meta: dict) -> dict:
    payload  = meta.get("payload", {}) or {}
    products = payload.get("products", {}) or {}
    return {
        "run_id":     meta.get("run_id"),
        "status":     _resolve_status(meta),
        "created_at": meta.get("created_at", ""),
        "updated_at": meta.get("updated_at", ""),
        "products":   list(products.keys()),
        "aoi_name":   payload.get("aoi_name", ""),
    }

def _ts_utc(ts) -> str:
    """Normalise any timestamp value to an ISO-8601 string with explicit UTC marker.

    Sources arrive in three formats:
      - DuckDB TIMESTAMP object  →  naive datetime (UTC internally)
      - datetime.utcfromtimestamp().isoformat()  →  'YYYY-MM-DDTHH:MM:SS' (no tz)
      - datetime.now(utc).isoformat()  →  '...+00:00'  (already correct)

    Without an explicit timezone marker JavaScript's Date() treats the string as
    *local* time, causing an offset against Snakemake's local-time log entries.
    """
    s = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    s = s.replace(" ", "T")          # DuckDB uses space separator
    if not (s.endswith("Z") or "+" in s[10:]):
        s += "Z"
    return s


def _run_to_detail(run_id: str, meta: dict) -> dict:
    payload = meta.get("payload", {}) or {}
    try:
        with _duckdb_connect() as conn:
            status_events = conn.execute(
                """SELECT event_time, event_type, message, payload_json
                   FROM run_events WHERE run_id=? ORDER BY event_time""",
                [run_id],
            ).fetchall()
            job_events = conn.execute(
                """SELECT started_at, finished_at, product, band, time_chunk, status, error
                   FROM jobs
                   WHERE run_id=?
                     AND (started_at IS NOT NULL OR finished_at IS NOT NULL)
                   ORDER BY COALESCE(finished_at, started_at)""",
                [run_id],
            ).fetchall()
    except Exception:
        status_events, job_events = [], []

    # Merge status events into timeline and build shelved set in one pass.
    merged = []
    shelved_chunks: set[tuple] = set()
    for ts, evtype, msg, payload_json in status_events:
        merged.append({"ts": _ts_utc(ts), "level": evtype, "msg": msg or evtype})
        if evtype == "job_shelved":
            try:
                d = json.loads(payload_json or "{}")
                shelved_chunks.add((d["prod"], d["band"], d["chunk"]))
            except Exception:
                pass

    # Track which jobs already have a DB-sourced finished event to avoid duplicates
    db_finished: set[tuple] = set()
    for started, finished, prod, band, chunk, jstatus, error in job_events:
        if started:
            merged.append({
                "ts":    _ts_utc(started),
                "level": "job_start",
                "msg":   f"Started {prod}/{band} [{chunk}]",
            })
        if finished:
            is_shelved = (prod, band, chunk) in shelved_chunks
            if is_shelved:
                # The job_shelved run_event carries the real message; skip a duplicate done entry.
                pass
            else:
                level = "job_done" if jstatus == "done" else "job_error"
                msg   = f"{'Finished' if jstatus == 'done' else 'Failed'} {prod}/{band} [{chunk}]"
                if error:
                    msg += f": {error}"
                merged.append({"ts": _ts_utc(finished), "level": level, "msg": msg})
            db_finished.add((prod, band, chunk))

    # Fill in finished events from the filesystem for jobs the log handler missed.
    # The parquet chunk file's mtime is the authoritative completion timestamp.
    chunks_dir   = RUNS_DIR / run_id / "intermediate" / "chunks"
    results_dir  = RUNS_DIR / run_id / "results"
    products_cfg = payload.get("products", {}) or {}
    for prod, cfg in products_cfg.items():
        for band in cfg.get("bands", []):
            for chunk in cfg.get("time_chunks", []):
                if (prod, band, chunk) in db_finished:
                    continue
                if (prod, band, chunk) in shelved_chunks:
                    continue
                parquet = chunks_dir / prod / f"{band}_{chunk}.parquet"
                if parquet.exists():
                    mtime = parquet.stat().st_mtime
                    ts    = _ts_utc(datetime.utcfromtimestamp(mtime))
                    merged.append({
                        "ts":    ts,
                        "level": "job_done",
                        "msg":   f"Finished {prod}/{band} [{chunk}]",
                    })

    # Add merge finished events from filesystem.
    # Track merge products that already have a finished event in run_events to avoid duplicates.
    merge_finished_prods = {
        e["msg"].split()[1]  # "Finished {prod} merge" → prod
        for e in merged
        if e.get("level") == "job_done" and e.get("msg", "").endswith(" merge")
    }
    for prod in products_cfg:
        if prod in merge_finished_prods:
            continue
        prod_result_dir = results_dir / prod
        if prod_result_dir.exists():
            parquet_files = [p for p in prod_result_dir.glob("*.parquet")]
            if parquet_files:
                newest = max(parquet_files, key=lambda p: p.stat().st_mtime)
                mtime  = newest.stat().st_mtime
                ts     = _ts_utc(datetime.utcfromtimestamp(mtime))
                merged.append({
                    "ts":    ts,
                    "level": "job_done",
                    "msg":   f"Finished {prod} merge",
                })

    merged.sort(key=lambda e: e["ts"])

    pid_file = RUNS_DIR / run_id / "logs" / "build_partial.pid"
    partial_build_running = False
    if pid_file.exists():
        try:
            partial_build_running = _is_pid_alive(int(pid_file.read_text().strip()))
        except Exception:
            pass

    return {
        **_run_to_summary(meta),
        "pid":                  meta.get("snakemake_pid"),
        "run_dir":              str(RUNS_DIR / run_id),
        "config":               payload,
        "job_counts":           _get_job_counts(run_id, meta),
        "events":               merged,
        "gee_concurrency":      int(payload.get("gee_concurrency") or DEFAULT_GEE_CONCURRENCY),
        "finished_products":    _list_finished_products(run_id),
        "partial_build_running": partial_build_running,
    }

# ─── Routes: GEE key ─────────────────────────────────────────────────────────

@app.get("/api/gee-key")
def gee_key_status():
    if not GEE_KEY_PATH.exists():
        return {"valid": False, "email": None, "error": "No key uploaded"}
    try:
        data  = json.loads(GEE_KEY_PATH.read_text())
        email = _validate_gee_key(data)
        return {"valid": True, "email": email, "error": None}
    except Exception as e:
        return {"valid": False, "email": None, "error": str(e)}

@app.post("/api/gee-key")
async def upload_gee_key(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data  = json.loads(content)
        email = _validate_gee_key(data)
    except ValueError as e:
        return {"valid": False, "email": None, "error": str(e)}
    except Exception:
        return {"valid": False, "email": None, "error": "Not valid JSON"}
    GEE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    GEE_KEY_PATH.write_bytes(content)
    return {"valid": True, "email": email, "error": None}

# ─── Routes: Products ─────────────────────────────────────────────────────────

def _build_products_response() -> list:
    result = []
    for pid, info in PRODUCT_REGISTRY.items():
        bands = [
            {"name": k, "description": k, "default_stats": v["default_stats"], "available_stats": v["stats"]}
            for k, v in info["content"].items()
        ]
        result.append({
            "id":           pid,
            "label":        info["label"],
            "description":  info["description"],
            "date_min":     info["min_date"],
            "date_max":     info["max_date"],
            "resolution_m": info["resolution_m"],
            "cadence":      info["cadence"],
            "categorical":  info["categorical"],
            "bands":        bands,
            "supported_stats": sorted({s for b in info["content"].values() for s in b["stats"]}),
        })
    return result

_PRODUCTS_RESPONSE: list = _build_products_response()

@app.get("/api/products")
def get_products():
    return _PRODUCTS_RESPONSE

# ─── Routes: Runs ─────────────────────────────────────────────────────────────

@app.get("/api/events")
def list_events(limit: int = 50):
    try:
        with _duckdb_connect() as conn:
            rows = conn.execute(
                """SELECT event_time, run_id, event_type, message
                   FROM run_events
                   ORDER BY event_time DESC
                   LIMIT ?""",
                [limit],
            ).fetchall()
    except Exception:
        rows = []
    return [
        {"ts": _ts_utc(ts), "run_id": run_id, "level": evtype, "msg": msg}
        for ts, run_id, evtype, msg in rows
    ]

@app.get("/api/runs")
def list_runs():
    return [_run_to_summary(m) for m in _list_saved_runs()]

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    meta = _load_yaml(run_id)
    if meta is None:
        raise HTTPException(404, "Run not found")
    return _run_to_detail(run_id, meta)

@app.post("/api/runs")
def submit_run(body: SubmitRunRequest):
    # Reject if any run is already active
    active = next(
        (m for m in _list_saved_runs() if _resolve_status(m) == "running"),
        None,
    )
    if active:
        raise HTTPException(409, f"Run '{active['run_id']}' is already running. Wait for it to finish or stop it first.")

    run_id  = re.sub(r"[^A-Za-z0-9]", "", (body.run_id or "").strip())[:40]
    if not run_id:
        run_id = "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))

    run_dir = RUNS_DIR / run_id
    if (run_dir / "run.yaml").exists():
        raise HTTPException(409, f"Run '{run_id}' already exists. Choose a different run ID.")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Find the AOI file saved under inputs/
    input_dir = run_dir / "inputs"
    aoi_files = sorted(input_dir.iterdir()) if input_dir.exists() else []
    # Prefer .shp, then .geojson/.parquet
    shp_file  = next((f for f in aoi_files if f.suffix == ".shp"), None)
    aoi_file  = shp_file or next(iter(aoi_files), None)
    if aoi_file is None:
        raise HTTPException(400, "No AOI uploaded for this run. Upload an AOI first.")

    # Build Snakemake payload (dict-of-dicts, matching main.py format)
    product_tasks: dict = {}
    for pc in body.products:
        info       = PRODUCT_REGISTRY.get(pc.product)
        if info is None:
            raise HTTPException(400, f"Unknown product: {pc.product}")
        cadence    = info["cadence"]
        time_chunks = get_time_chunks(pc.date_start, pc.date_end, cadence)
        # Clamp end date to the last day of the end month for monthly cadence
        dt_end     = datetime.strptime(pc.date_end, "%Y-%m-%d")
        dt_end     = dt_end.replace(day=calendar.monthrange(dt_end.year, dt_end.month)[1])

        product_tasks[pc.product] = {
            "ee_collection":    info.get("ee_collection"),
            "multi_collections": info.get("multi_collections"),
            "bands":            pc.bands,
            "statistics":       pc.stats,       # Snakefile uses "statistics"
            "scale":            info["scale"],
            "resolution_m":     info.get("resolution_m", info["scale"]),
            "cadence":          cadence,
            "categorical":         info["categorical"],
            "normalize_histogram": info.get("normalize_histogram", False),
            "start_date":       pc.date_start,
            "end_date":         dt_end.strftime("%Y-%m-%d"),
            "time_chunks":      time_chunks,
            "gee_weight":       info.get("gee_weight", 1),
            "tile_scale":       info.get("tile_scale", 1),
            # Per-band QA bit-mask configs (None for bands/products with no masking).
            "band_masks":       {
                band: info.get("content", {}).get(band, {}).get("qa_mask")
                for band in pc.bands
            },
            # Per-band unit transforms applied before reduction (e.g. Kelvin → Celsius).
            "band_transforms":  {
                band: info.get("content", {}).get(band, {}).get("band_transform")
                for band in pc.bands
            },
            # Per-band derived-band compute specs (e.g. LST_Mean from Day + Night).
            "band_computes":    {
                band: info.get("content", {}).get(band, {}).get("band_compute")
                for band in pc.bands
            },
        }

    payload = {
        "run_id":                run_id,
        "shp_path":              aoi_file.as_posix(),
        "products":              product_tasks,
        "output_dir":            _results_dir(run_id).as_posix(),
        "aoi_name":              aoi_file.name,
        "app_dir":               APP_DIR.as_posix(),
        "gee_concurrency":       max(1, body.gee_concurrency),
        "id_column":             body.id_column or "",
    }

    # Register run (status=queued → running)
    _update_registry(run_id, payload, status="queued")
    _update_registry(run_id, payload, status="running", bump_attempt=True)

    # Pre-populate jobs table so progress bar works immediately
    _initialise_jobs(run_id, payload)

    log_dir  = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "snakemake_run.log"
    with open(log_path, "a", encoding="utf-8") as lh:
        lh.write(f"\n[{datetime.now(timezone.utc).isoformat()}] API submit\n")

    _launch_snakemake(run_id, payload, log_path)
    return _run_to_detail(run_id, _load_yaml(run_id) or {})

@app.delete("/api/runs/{run_id}")
def stop_run(run_id: str):
    meta = _load_yaml(run_id)
    if meta is None:
        raise HTTPException(404, "Run not found")

    pid = meta.get("snakemake_pid")
    if pid and _is_pid_alive(pid):
        _signal_process_tree(int(pid), signal.SIGTERM)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and _is_pid_alive(pid):
            time.sleep(0.1)
        if _is_pid_alive(pid):
            _signal_process_tree(int(pid), _SIGKILL)

    payload = meta.get("payload", {})
    _update_registry(run_id, payload, status="stopped",
                     error_message="Stopped by user from React UI.")
    return {"ok": True}

@app.post("/api/runs/{run_id}/pause")
def pause_run(run_id: str):
    meta = _load_yaml(run_id)
    if meta is None:
        raise HTTPException(404, "Run not found")
    if _resolve_status(meta) != "running":
        raise HTTPException(409, "Run is not currently running.")

    pid = meta.get("snakemake_pid")
    if pid and _is_pid_alive(int(pid)):
        if sys.platform == "win32":
            # Windows has no SIGSTOP — kill the process instead.
            # Snakemake checkpoints completed jobs in .snakemake/, so resume
            # can restart it and it will skip already-finished work.
            _signal_process_tree(int(pid), signal.SIGTERM)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and _is_pid_alive(int(pid)):
                time.sleep(0.1)
            if _is_pid_alive(int(pid)):
                _signal_process_tree(int(pid), _SIGKILL)
        else:
            _signal_process_tree(int(pid), signal.SIGSTOP)

    payload = meta.get("payload", {})
    _update_registry(run_id, payload, status="paused", clear_pid=(sys.platform == "win32"))
    return {"ok": True}


@app.post("/api/runs/{run_id}/resume")
def resume_run(run_id: str, body: ResumeRunRequest = ResumeRunRequest()):
    meta = _load_yaml(run_id)
    if meta is None:
        raise HTTPException(404, "Run not found")
    if _resolve_status(meta) != "paused":
        raise HTTPException(409, "Run is not currently paused.")

    pid     = meta.get("snakemake_pid")
    payload = _fix_payload_paths(run_id, dict(meta.get("payload") or {}))

    stored_concurrency = int(payload.get("gee_concurrency") or DEFAULT_GEE_CONCURRENCY)
    new_concurrency    = max(1, body.gee_concurrency) if body.gee_concurrency is not None else None
    changing           = new_concurrency is not None and new_concurrency != stored_concurrency

    # On Windows, pause kills the process — resume must always restart it.
    if sys.platform == "win32":
        changing = True

    if changing:
        # Kill the frozen process tree and restart with the new concurrency.
        if new_concurrency is not None:
            payload["gee_concurrency"] = new_concurrency
        if pid and _is_pid_alive(int(pid)):
            if sys.platform != "win32":
                _signal_process_tree(int(pid), signal.SIGCONT)   # unfreeze first so SIGTERM is handled
                time.sleep(0.2)
            _signal_process_tree(int(pid), signal.SIGTERM)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and _is_pid_alive(int(pid)):
                time.sleep(0.1)
            if _is_pid_alive(int(pid)):
                _signal_process_tree(int(pid), _SIGKILL)

        run_dir  = RUNS_DIR / run_id
        log_dir  = run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "snakemake_run.log"
        with open(log_path, "a", encoding="utf-8") as lh:
            lh.write(f"\n[{datetime.now(timezone.utc).isoformat()}] API resume (new gee={new_concurrency})\n")
        _update_registry(run_id, payload, status="running", bump_attempt=True)
        _launch_snakemake(run_id, payload, log_path)
    else:
        # Simple SIGCONT — resume exactly where we left off.
        # Do NOT call _update_registry here: it would clear snakemake_pid (by design
        # for restarts) and _resolve_status would then flip the run to "failed" after
        # the 60 s grace period. Instead, patch only the status field in-place.
        if pid and _is_pid_alive(int(pid)) and sys.platform != "win32":
            _signal_process_tree(int(pid), signal.SIGCONT)
        existing = _load_yaml(run_id) or {}
        existing["status"]     = "running"
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        _save_yaml(run_id, existing)
        _upsert_run_status(run_id, existing)
        _append_event(run_id, "status_change", "running", "Run resumed by user")

    return _run_to_detail(run_id, _load_yaml(run_id) or {})


@app.post("/api/runs/{run_id}/reset")
def reset_run(run_id: str):
    """Delete all generated outputs for a run, preserving run.yaml and inputs/."""
    meta = _load_yaml(run_id)
    if meta is None:
        raise HTTPException(404, "Run not found")

    if _resolve_status(meta) == "running":
        raise HTTPException(409, "Stop the run before resetting it.")

    run_dir = RUNS_DIR / run_id
    import shutil
    for subdir in ("intermediate", "results", "logs", ".snakemake"):
        target = run_dir / subdir
        if target.exists():
            shutil.rmtree(target)

    payload = meta.get("payload", {})
    _update_registry(run_id, payload, status="stopped",
                     error_message="Reset by user.")
    return {"ok": True}

@app.post("/api/runs/{run_id}/retry")
def retry_run(run_id: str, body: RetryRunRequest = RetryRunRequest()):
    meta = _load_yaml(run_id)
    if meta is None:
        raise HTTPException(404, "Run not found")

    if _resolve_status(meta) in ("running", "paused"):
        raise HTTPException(409, "Run is already active. Stop or pause before retrying.")

    active = next(
        (m for m in _list_saved_runs()
         if m.get("run_id") != run_id and _resolve_status(m) == "running"),
        None,
    )
    if active:
        raise HTTPException(409, f"Run '{active['run_id']}' is already running.")

    payload = _fix_payload_paths(run_id, dict(meta.get("payload") or {}))
    if body.gee_concurrency is not None:
        payload["gee_concurrency"] = max(1, body.gee_concurrency)

    run_dir  = RUNS_DIR / run_id
    log_dir  = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "snakemake_run.log"
    with open(log_path, "a", encoding="utf-8") as lh:
        lh.write(f"\n[{datetime.now(timezone.utc).isoformat()}] API retry\n")

    _update_registry(run_id, payload, status="running", bump_attempt=True)
    _launch_snakemake(run_id, payload, log_path)
    return _run_to_detail(run_id, _load_yaml(run_id) or {})

@app.get("/api/runs/{run_id}/log")
def get_run_log(run_id: str, lines: int = 100):
    log_path = RUNS_DIR / run_id / "logs" / "snakemake_run.log"
    if not log_path.exists():
        return {"lines": []}
    with open(log_path, "rb") as f:
        # Read only the last chunk needed rather than loading the whole file.
        chunk_size = lines * 200  # ~200 bytes per line on average
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(max(0, file_size - chunk_size))
        raw = f.read().decode("utf-8", errors="replace")
    all_lines = raw.splitlines()
    tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
    # If we seeked mid-file the first element may be a partial line; drop it only
    # when we didn't read from the start.
    if file_size > chunk_size and len(all_lines) > lines:
        tail = all_lines[1:][-lines:]
    return {"lines": tail}

@app.post("/api/runs/{run_id}/partial")
def trigger_partial(run_id: str):
    script   = APP_DIR / "scripts" / "build_partial.py"
    pid_file = RUNS_DIR / run_id / "logs" / "build_partial.pid"
    log_path = RUNS_DIR / run_id / "logs" / "build_partial.log"

    if pid_file.exists():
        try:
            if _is_pid_alive(int(pid_file.read_text().strip())):
                return {"ok": False, "running": True}
        except Exception:
            pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # CREATE_NEW_PROCESS_GROUP isolates the child from Ctrl-C / CTRL_C_EVENT
    # signals sent to the uvicorn process group on Windows, preventing a
    # console interrupt from crashing background build jobs and the server.
    _pgroup_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    with open(log_path, "a") as log_fh:
        proc = subprocess.Popen(
            [sys.executable, str(script), run_id, str(RUNS_DIR)],
            stdout=log_fh,
            stderr=log_fh,
            creationflags=_pgroup_flag,
        )
    pid_file.write_text(str(proc.pid))
    return {"ok": True}

def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise HTTPException(400, "Invalid filename")
    return name


def _safe_extract_zip(zf: zipfile.ZipFile, destination: Path):
    dest_root = destination.resolve()
    for member in zf.infolist():
        member_path = Path(member.filename)
        if member_path.is_absolute():
            raise HTTPException(400, "Zip contains invalid entry paths")
        target = (destination / member_path).resolve(strict=False)
        if not target.is_relative_to(dest_root):
            raise HTTPException(400, "Zip contains invalid entry paths")
    zf.extractall(destination)


# ─── Routes: AOI upload ───────────────────────────────────────────────────────

def _process_aoi(content: bytes, ext: str, dest: Path, input_dir: Path):
    if ext == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            zp = Path(tmp) / "aoi.zip"
            zp.write_bytes(content)
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(tmp)
            shp_files = list(Path(tmp).glob("**/*.shp"))
            if not shp_files:
                raise HTTPException(400, "No .shp found in zip")
            # Also extract into input_dir for persistence
            with zipfile.ZipFile(dest) as zf:
                _safe_extract_zip(zf, input_dir)
            gdf = gpd.read_file(shp_files[0])
    elif ext in (".geojson", ".json"):
        gdf = gpd.read_file(io.BytesIO(content))
    elif ext in (".parquet", ".geoparquet"):
        gdf = gpd.read_parquet(io.BytesIO(content))
    else:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    if gdf.empty or gdf.geometry.isna().all():
        raise HTTPException(400, "AOI contains no valid geometries")

    gdf_4326 = gdf.to_crs(epsg=4326)
    bounds   = gdf_4326.total_bounds.tolist()
    # Simplify preview geometry so serialisation is fast even for complex files.
    # tolerance=0.001° ≈ 100m — fine enough for a map thumbnail.
    preview_gdf = gdf_4326.head(200).copy()
    preview_gdf["geometry"] = preview_gdf.geometry.simplify(0.001, preserve_topology=True)
    preview  = json.loads(preview_gdf.to_json())

    # Introspect non-geometry columns so the frontend can offer an ID column picker.
    geom_col  = gdf.geometry.name
    data_cols = [c for c in gdf.columns if c != geom_col]
    column_samples: dict = {}
    column_has_duplicates: dict = {}
    for col in data_cols:
        non_null = gdf[col].dropna()
        column_samples[col]       = [str(v) for v in non_null.head(3).tolist()]
        column_has_duplicates[col] = bool(gdf[col].duplicated().any())

    return {
        "feature_count":          len(gdf),
        "crs":                    str(gdf.crs),
        "bounds":                 bounds,
        "geojson_preview":        preview,
        "columns":                data_cols,
        "column_samples":         column_samples,
        "column_has_duplicates":  column_has_duplicates,
    }


@app.post("/api/runs/{run_id}/aoi")
async def upload_aoi(run_id: str, file: UploadFile = File(...)):
    import asyncio
    run_id    = re.sub(r"[^A-Za-z0-9]", "", run_id)[:40]
    run_dir   = RUNS_DIR / run_id
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    content  = await file.read()
    filename = _sanitize_filename(file.filename or "aoi")
    ext      = Path(filename).suffix.lower()
    dest     = input_dir / filename
    dest.write_bytes(content)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _process_aoi, content, ext, dest, input_dir)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to read AOI: {e}")

    return result

# ─── Routes: Downloads ────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/download/{product}")
def download_parquet(run_id: str, product: str):
    path = _find_product_parquet(run_id, product)
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )

@app.get("/api/runs/{run_id}/download/{product}/csv")
def download_csv(run_id: str, product: str):
    import pyarrow.parquet as pq
    from starlette.background import BackgroundTask
    path     = _find_product_parquet(run_id, product)
    schema   = pq.read_schema(path)
    geo      = schema.metadata.get(b"geo") if schema.metadata else None
    geom_col = json.loads(geo).get("primary_column", "geometry") if geo else "geometry"
    non_geom = [c for c in schema.names if c != geom_col]
    cols_sql = ", ".join(f'"{c}"' for c in non_geom)

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp_path = tmp.name
    tmp.close()

    with duckdb.connect(":memory:") as conn:
        conn.execute(
            f"COPY (SELECT {cols_sql} FROM read_parquet('{path.as_posix()}')) "
            f"TO '{Path(tmp_path).as_posix()}' (HEADER, DELIMITER ',')"
        )

    return FileResponse(
        tmp_path,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{path.stem}.csv"'},
        background=BackgroundTask(os.unlink, tmp_path),
    )

@app.get("/api/runs/{run_id}/download/{product}/partial-csv")
def download_partial_csv(run_id: str, product: str):
    import pyarrow.parquet as pq
    from starlette.background import BackgroundTask
    partial_dir = _results_dir(run_id) / "partial_checkout" / product
    files       = sorted(partial_dir.glob("*.parquet")) if partial_dir.exists() else []
    if not files:
        raise HTTPException(404, "No partial checkout file yet")
    path     = files[-1]
    schema   = pq.read_schema(path)
    geo      = schema.metadata.get(b"geo") if schema.metadata else None
    geom_col = json.loads(geo).get("primary_column", "geometry") if geo else "geometry"
    non_geom = [c for c in schema.names if c != geom_col]
    cols_sql = ", ".join(f'"{c}"' for c in non_geom)

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    tmp_path = tmp.name
    tmp.close()

    with duckdb.connect(":memory:") as conn:
        conn.execute(
            f"COPY (SELECT {cols_sql} FROM read_parquet('{path.as_posix()}')) "
            f"TO '{Path(tmp_path).as_posix()}' (HEADER, DELIMITER ',')"
        )

    return FileResponse(
        tmp_path,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{path.stem}.csv"'},
        background=BackgroundTask(os.unlink, tmp_path),
    )

@app.get("/api/runs/{run_id}/download/{product}/partial")
def download_partial_parquet(run_id: str, product: str):
    partial_dir = _results_dir(run_id) / "partial_checkout" / product
    files       = sorted(partial_dir.glob("*.parquet")) if partial_dir.exists() else []
    if not files:
        raise HTTPException(404, "No partial checkout file yet")
    path = files[-1]
    return FileResponse(
        str(path),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
    )

# ─── SPA static files (pixi / local dev only) ─────────────────────────────────
# Mount the built React app so a single `uvicorn` process serves everything.
# In Docker the nginx container handles this instead, so this is a no-op there.
_dist = APP_DIR / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="spa")
