# ============================================================
# XGBoost rules — Indonesia & OpenDengue
# Invoke:
#   snakemake reports/tables/<name>/<region>_results.csv \
#       --configfile config/<dataset>/xgboost_<experiment>.yaml \
#       --cores 4 --resources gpu=1
#
# Scripts are invoked via CLI; snakemake.config is not used.
# All experiment parameters are read from the configfile by the scripts.
# ============================================================

import pandas as pd

_name    = config.get("name", "")
_dataset = config.get("dataset", "")
_regions = config.get("regions", [])

_tune_script  = (
    "src/Indonesia/XGBoost/tune.py"
    if _dataset == "indonesia"
    else "src/OpenDengue/XGBoost/tune_xgboost.py"
)
_train_script = (
    "src/Indonesia/XGBoost/train.py"
    if _dataset == "indonesia"
    else "src/OpenDengue/XGBoost/train_xgboost.py"
)
_eval_script  = (
    "src/Indonesia/XGBoost/eval.py"
    if _dataset == "indonesia"
    else "src/OpenDengue/XGBoost/eval_xgboost.py"
)


rule tune_xgboost:
    input:
        data = config.get("data", {}).get("path", ""),
    output:
        params_csv = f"reports/tables/{_name}/{{region}}_params.csv",
    params:
        cfg    = lambda wc: workflow.configfiles[-1],
        script = _tune_script,
    resources:
        gpu = 1,
    shell:
        "python {params.script} --config {params.cfg} --region {wildcards.region}"


rule train_xgboost:
    input:
        data       = config.get("data", {}).get("path", ""),
        params_csv = f"reports/tables/{_name}/{{region}}_params.csv",
    output:
        model = f"models/xgboost/{_name}/{{region}}.json",
    params:
        cfg    = lambda wc: workflow.configfiles[-1],
        script = _train_script,
    resources:
        gpu = 1,
    shell:
        "python {params.script} --config {params.cfg} --region {wildcards.region} --params {input.params_csv}"


rule eval_xgboost:
    input:
        data  = config.get("data", {}).get("path", ""),
        model = f"models/xgboost/{_name}/{{region}}.json",
    output:
        results_csv = f"reports/tables/{_name}/{{region}}_results.csv",
    params:
        cfg    = lambda wc: workflow.configfiles[-1],
        script = _eval_script,
    resources:
        gpu = 1,
    shell:
        "python {params.script} --config {params.cfg} --region {wildcards.region} --model {input.model}"
