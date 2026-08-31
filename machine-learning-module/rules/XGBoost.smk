# ============================================================
# XGBoost rules — Indonesia & OpenDengue
# Invoke (from anywhere — paths below are anchored to PIPELINE_ROOT /
# PROJECT_ROOT, not to the invocation cwd):
#   snakemake -s machine-learning-module/Snakefile \
#       "<PIPELINE_ROOT>/results/XGBoost/<name>/<region>/results.csv" \
#       --configfile machine-learning-module/config/OpenDengue/xgboost_<experiment>.yaml \
#       --cores 4 --resources gpu=1
#
# Scripts are invoked via CLI; snakemake.config is not used.
# All experiment parameters are read from the configfile by the scripts.
#
# OpenDengue outputs are all written under machine-learning-module/results/XGBoost/{name}/{region}/,
# mirroring results/STGNN/{name}/. Indonesia outputs keep the legacy
# reports/tables + models/xgboost layout, also under PIPELINE_ROOT.
# ============================================================

import pandas as pd

_name      = config.get("name", "")
_dataset   = config.get("dataset", "")
_data_path = config.get("data", {}).get("path", "")
_data_path = str(PROJECT_ROOT / _data_path) if _data_path else ""

if _dataset == "indonesia":
    _params_csv  = str(PIPELINE_ROOT / "reports" / "tables" / _name / "{region}_params.csv")
    _model_out   = str(PIPELINE_ROOT / "models" / "xgboost" / _name / "{region}.json")
    _results_csv = str(PIPELINE_ROOT / "reports" / "tables" / _name / "{region}_results.csv")
    _regions     = config.get("regions", [])
else:
    _region_dir  = PIPELINE_ROOT / "results" / "XGBoost" / _name / "{region}"
    _params_csv  = str(_region_dir / "params.csv")
    _model_out   = str(_region_dir / "model.json")
    _results_csv = str(_region_dir / "results.csv")

    # Regions default to the unique values of the config's region_column
    # (falling back to unit_column) so `all_xgboost` can expand over every
    # country in the dataset without listing them by hand in the yaml.
    _regions = config.get("regions")
    if not _regions and _data_path and Path(_data_path).exists():
        _region_col = (config.get("region_column")
                       or config.get("data", {}).get("region_column")
                       or config.get("unit_column")
                       or config.get("data", {}).get("unit_column", "name"))
        _regions = sorted(pd.read_csv(_data_path, usecols=[_region_col])[_region_col].dropna().unique().tolist())
    _regions = _regions or []

_tune_script  = str(Path(workflow.basedir) / (
    "src/Indonesia/XGBoost/tune.py"
    if _dataset == "indonesia"
    else "src/OpenDengue/XGBoost/tune_xgboost.py"
))
_train_script = str(Path(workflow.basedir) / (
    "src/Indonesia/XGBoost/train.py"
    if _dataset == "indonesia"
    else "src/OpenDengue/XGBoost/train_xgboost.py"
))
_eval_script  = str(Path(workflow.basedir) / (
    "src/Indonesia/XGBoost/eval.py"
    if _dataset == "indonesia"
    else "src/OpenDengue/XGBoost/eval_xgboost.py"
))


rule tune_xgboost:
    input:
        data = _data_path,
    output:
        params_csv = _params_csv,
    params:
        cfg    = lambda wc: workflow.configfiles[-1],
        script = _tune_script,
    resources:
        gpu = 1,
    shell:
        'python {params.script} --config "{params.cfg}" --region "{wildcards.region}"'


rule train_xgboost:
    input:
        data       = _data_path,
        params_csv = _params_csv,
    output:
        model = _model_out,
    params:
        cfg    = lambda wc: workflow.configfiles[-1],
        script = _train_script,
    resources:
        gpu = 1,
    shell:
        'python {params.script} --config "{params.cfg}" --region "{wildcards.region}" --params "{input.params_csv}"'


rule eval_xgboost:
    input:
        data  = _data_path,
        model = _model_out,
    output:
        results_csv = _results_csv,
    params:
        cfg    = lambda wc: workflow.configfiles[-1],
        script = _eval_script,
    resources:
        gpu = 1,
    shell:
        'python {params.script} --config "{params.cfg}" --region "{wildcards.region}" --model "{input.model}"'


rule all_xgboost:
    input:
        expand(_results_csv, region=_regions),
