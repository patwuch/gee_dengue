"""
Snakemake workflow for GEE extraction using GeoJSON → GeoParquet pipeline.

Workflow stages:
1. Extract zonal stats from GEE as GeoJSON (preserves geometry)
2. Convert GeoJSON chunks to GeoParquet using DuckDB
3. Merge GeoParquet chunks into final product files

Cloud-ready: GeoParquet is optimized for object storage (S3, GCS, Azure)
"""
import pandas as pd
import os
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google")

# Ensure the project root is on sys.path for scripts run by Snakemake
_project_root = str(Path(workflow.basedir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.environ.setdefault("PYTHONPATH", _project_root)

from workflow.time_chunks import infer_time_chunks, chunk_start_date, chunk_end_date
from workflow.state import update_run_state


# Configuration
PRODUCTS = config.get("products", {})
SHP = config.get("shp_path", "")
RUN_ID = config.get("run_id", "default")
APP_DIR = config.get("app_dir", "/app")
ID_COLUMN = config.get("id_column", "")
_gee_slots  = int(config.get("gee_concurrency", 10))
# Max gee_weight across selected products — determines true concurrent chunk limit.
# A product with gee_weight=5 can only run floor(10/5)=2 chunks at once.
_max_weight = max((p.get("gee_weight", 1) for p in PRODUCTS.values()), default=1)
_max_concurrent = _gee_slots // _max_weight
# Window = how many chunks ahead per band Snakemake is allowed to unlock.
# Must be >= _max_concurrent so the GEE pool can stay fully saturated even when
# only one product/band is running. Dividing by _num_products was wrong: it
# collapsed to 1 once products finished, serialising the remaining work.
# The gee resource constraint is the sole concurrency limiter — the window
# just needs to be large enough not to artificially starve it.
CHAIN_PARALLEL_WINDOW = int(config.get("chain_parallel_window", _max_concurrent))
# Finest sensor resolution among all selected products — used to drive AOI simplification.
# Preprocessing simplifies to max(resolution_tol, budget_tol); workers re-simplify
# further if their product's native scale is coarser than the finest sensor.
_finest_resolution_m = min((p.get("resolution_m", 30) for p in PRODUCTS.values()), default=30)



# Directory structure — absolute paths so the Snakefile works correctly
# regardless of which per-run directory Snakemake uses as its working directory.
GEOJSON_CHUNKS_DIR = f"{APP_DIR}/data/runs/{RUN_ID}/intermediate/geojson"
PARQUET_CHUNKS_DIR = f"{APP_DIR}/data/runs/{RUN_ID}/intermediate/chunks"
RESULTS_DIR        = f"{APP_DIR}/data/runs/{RUN_ID}/results"
LOGS_DIR           = f"{APP_DIR}/data/runs/{RUN_ID}/logs"
PREPPED_AOI        = f"{APP_DIR}/data/runs/{RUN_ID}/intermediate/aoi_prepped.parquet"


def get_previous_chunk_output(wildcards):
    """
    Enforce time-series ordering within each band using a sliding window.
    Chunk N cannot start until chunk N-CHAIN_PARALLEL_WINDOW's parquet exists,
    ensuring partial checkouts always contain a contiguous time series from
    the start rather than scattered chunks.
    Uses ancient() so a re-run of the prev chunk (e.g. after a restart) does
    not cascade mtime invalidations through the rest of the chain.
    """
    chunks = infer_time_chunks(PRODUCTS[wildcards.prod])
    try:
        idx = chunks.index(wildcards.time_chunk)
    except ValueError:
        return []

    if idx < CHAIN_PARALLEL_WINDOW:
        return []

    prev_chunk = chunks[idx - CHAIN_PARALLEL_WINDOW]
    return ancient(f"{PARQUET_CHUNKS_DIR}/{wildcards.prod}/{wildcards.band}_{prev_chunk}.parquet")


def get_final_targets():
    """Define final output files: merged GeoParquet per product."""
    targets = []
    for prod, settings in PRODUCTS.items():
        start, end = settings["start_date"], settings["end_date"]
        targets.append(f"{RESULTS_DIR}/{prod}/{prod}_{start}_to_{end}.parquet")
    return targets


# ==============================================================================
# Rules
# ==============================================================================

rule all:
    input:
        get_final_targets()

onsuccess:
    from workflow.state import update_run_state, write_run_warnings_summary
    update_run_state(
        run_yaml = f"{APP_DIR}/data/runs/{RUN_ID}/run.yaml",
        db_path  = f"{APP_DIR}/data/runs/run_state.duckdb",
        run_id   = RUN_ID,
        status   = "completed",
        message  = "Run completed successfully"
    )
    write_run_warnings_summary(
        db_path   = f"{APP_DIR}/data/runs/run_state.duckdb",
        run_id    = RUN_ID,
        runs_dir  = f"{APP_DIR}/data/runs",
    )

onerror:
    from workflow.state import update_run_state
    update_run_state(
        run_yaml = f"{APP_DIR}/data/runs/{RUN_ID}/run.yaml",
        db_path  = f"{APP_DIR}/data/runs/run_state.duckdb",
        run_id   = RUN_ID,
        status   = "failed",
        message  = "Run failed"
    )

rule preprocess_aoi:
    """
    Pre-process the AOI to the largest scale it can get away with based on the products selected for that run
    A second 
    """
    input:
        shp = SHP
    output:
        aoi = PREPPED_AOI
    params:
        finest_resolution_m = _finest_resolution_m,
        id_column           = ID_COLUMN
    log:
        f"{LOGS_DIR}/preprocess_aoi.log"
    script:
        "scripts/preprocess_aoi.py"


rule extract_geojson_chunk:
    """
    Step 1: Extract zonal statistics from GEE as GeoJSON.
    One job = One time chunk + One Product + One Band.
    Preserves geometry for spatial analysis.
    """
    input:
        aoi  = PREPPED_AOI,
        prev = get_previous_chunk_output
    output:
        geojson = temp(f"{GEOJSON_CHUNKS_DIR}/{{prod}}/{{band}}_{{time_chunk}}.geojson")
    resources:
        gee=lambda wildcards: PRODUCTS[wildcards.prod].get("gee_weight", 1)
    retries: 2  # matches GEE_TIMEOUT_MAX_RETRIES-1; on 3rd timeout the worker shelves the chunk
    threads: 1
    wildcard_constraints:
        # Band names are plain word identifiers — no date-like suffixes — so greedy
        # matching cannot consume _YYYY-MM into {band} instead of {time_chunk}.
        band       = r"[A-Za-z][A-Za-z0-9_]*",
        # YYYY (annual) | YYYY-MM_YYYY-MM (3-month batch) | YYYY-MM (legacy single-month)
        time_chunk = r"\d{4}(-\d{2}(_\d{4}-\d{2})?)?"
    params:
        stats = lambda wildcards: PRODUCTS[wildcards.prod]["statistics"],
        ee_collection = lambda wildcards: PRODUCTS[wildcards.prod].get("ee_collection"),
        multi_collections = lambda wildcards: PRODUCTS[wildcards.prod].get("multi_collections"),
        scale = lambda wildcards: PRODUCTS[wildcards.prod]["scale"],
        tile_scale = lambda wildcards: PRODUCTS[wildcards.prod].get("tile_scale", 1),
        cadence = lambda wildcards: PRODUCTS[wildcards.prod].get("cadence", "monthly"),
        categorical          = lambda wildcards: PRODUCTS[wildcards.prod].get("categorical", False),
        normalize_histogram  = lambda wildcards: PRODUCTS[wildcards.prod].get("normalize_histogram", False),
        qa_mask = lambda wildcards: PRODUCTS[wildcards.prod].get("band_masks", {}).get(wildcards.band),
        band_transform = lambda wildcards: PRODUCTS[wildcards.prod].get("band_transforms", {}).get(wildcards.band),
        band_compute   = lambda wildcards: PRODUCTS[wildcards.prod].get("band_computes",   {}).get(wildcards.band),
        start_date = lambda wc: chunk_start_date(wc.time_chunk),
        end_date   = lambda wc: chunk_end_date(wc.time_chunk),
        finest_resolution_m = _finest_resolution_m
    log:
        f"{LOGS_DIR}/{{prod}}/{{band}}_{{time_chunk}}_geojson.log"
    script:
        "scripts/worker_geojson.py"

rule convert_to_parquet:
    """
    Step 2: Convert GeoJSON chunk to GeoParquet using DuckDB.
    Applies compression and columnar storage optimization.
    """
    input:
        geojson = f"{GEOJSON_CHUNKS_DIR}/{{prod}}/{{band}}_{{time_chunk}}.geojson"
    output:
        parquet = temp(f"{PARQUET_CHUNKS_DIR}/{{prod}}/{{band}}_{{time_chunk}}.parquet")
    threads: 1
    wildcard_constraints:
        band       = r"[A-Za-z][A-Za-z0-9_]*",
        time_chunk = r"\d{4}(-\d{2}(_\d{4}-\d{2})?)?"
    log:
        f"{LOGS_DIR}/{{prod}}/{{band}}_{{time_chunk}}_parquet.log"
    script:
        "scripts/geojson_to_parquet.py"

rule merge_band_chunks:
    """
    Step 3a: Merge all time chunks for a single band into one parquet.
    Pure row-append (LONG merge) — identical schema per band, so DuckDB streams
    without pivoting. Memory cost scales with one band's data, not all bands.
    """
    input:
        chunks = lambda wildcards: [
            f"{PARQUET_CHUNKS_DIR}/{wildcards.prod}/{wildcards.band}_{chunk}.parquet"
            for chunk in infer_time_chunks(PRODUCTS[wildcards.prod])
        ]
    output:
        band_parquet = temp(f"{PARQUET_CHUNKS_DIR}/{{prod}}/merged_{{band}}.parquet")
    params:
        merge_strategy = "long",
        band = lambda wildcards: wildcards.band
    threads: 1
    wildcard_constraints:
        band = r"[A-Za-z][A-Za-z0-9_]*"
    log:
        f"{LOGS_DIR}/{{prod}}/merge_band_{{band}}.log"
    script:
        "scripts/merge_parquet.py"


rule merge_product_parquet:
    """
    Step 3b: Merge per-band parquets into a single wide product file.
    Receives one already-complete parquet per band (not N_bands × N_chunks),
    so the wide pivot operates on at most N_band files instead of N_bands × N_chunks.
    """
    input:
        chunks = lambda wildcards: [
            f"{PARQUET_CHUNKS_DIR}/{wildcards.prod}/merged_{band}.parquet"
            for band in PRODUCTS[wildcards.prod]["bands"]
        ],
        aoi = PREPPED_AOI
    output:
        merged = f"{RESULTS_DIR}/{{prod}}/{{prod}}_{{start}}_to_{{end}}.parquet"
    params:
        merge_strategy = "wide"
    threads: 2
    log:
        f"{LOGS_DIR}/{{prod}}/merge_{{start}}_to_{{end}}.log"
    script:
        "scripts/merge_parquet.py"
