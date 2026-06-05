# ============================================================
# Random Forest rules — OpenDengue
# Invoke:
#   snakemake reports/tables/<name>/<region>_rf_results.csv \
#       --configfile config/opendengue/randomforest_<experiment>.yaml \
#       --cores 4
# ============================================================

_name    = config.get("name", "")
_regions = config.get("regions", [])


rule tune_rf:
    input:
        data = config.get("data", {}).get("path", ""),
    output:
        params_csv = f"reports/tables/{_name}/{{region}}_rf_params.csv",
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    threads: 4,
    shell:
        "python src/OpenDengue/RandomForest/tune_RF.py --config {params.cfg} --region {wildcards.region}"


rule train_rf:
    input:
        data       = config.get("data", {}).get("path", ""),
        params_csv = f"reports/tables/{_name}/{{region}}_rf_params.csv",
    output:
        model = f"models/rf/{_name}/{{region}}.joblib",
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    threads: 4,
    shell:
        "python src/OpenDengue/RandomForest/train_RF.py --config {params.cfg} --region {wildcards.region} --params {input.params_csv} --n-jobs {threads}"


rule eval_rf:
    input:
        data  = config.get("data", {}).get("path", ""),
        model = f"models/rf/{_name}/{{region}}.joblib",
    output:
        results_csv = f"reports/tables/{_name}/{{region}}_rf_results.csv",
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    shell:
        "python src/OpenDengue/RandomForest/eval_RF.py --config {params.cfg} --region {wildcards.region} --model {input.model}"
