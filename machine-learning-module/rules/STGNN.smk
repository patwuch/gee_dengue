# ============================================================
# STGNN rules — OpenDengue SEA
# Scripts: src/OpenDengue/STGNN/
#
# Invoke:
#   snakemake results/STGNN/<name>/metrics.json \
#       --configfile config/<experiment>.yaml \
#       --cores 4 --resources gpu=1
#
# Scripts receive all experiment parameters via snakemake.config.
# ============================================================

_name = config.get("name", "")


_window_sizes = (
    config.get("tune", {})
          .get("search_space", {})
          .get("stgnn", {})
          .get("window_size", {})
          .get("choices", [0])
)



rule preprocess_stgnn:
    message:
        "Preprocessing for STGNN with experiment name '{_name}' and window sizes {_window_sizes}."
    input:
        MERGED_DENGUE_ENV_DATA,
    output:
        tensors = expand(
            "data/processed/STGNN/{name}/window_{w}/tensors.pt",
            name=_name,
            w=_window_sizes,
        ),
        scaler  = f"data/processed/STGNN/{_name}/scaler.pkl",
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    script:
        f"{workflow.basedir}/src/OpenDengue/STGNN/preprocess/pipeline.py"

rule tune_stgnn:
    input:
        tensors = expand(
            "data/processed/STGNN/{name}/window_{w}/tensors.pt",
            name = _name,
            w    = _window_sizes,
        ),
        scaler  = f"data/processed/STGNN/{_name}/scaler.pkl",
    output:
        best_params = f"results/STGNN/{_name}/best_params.json",
    resources:
        gpu = 1,
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    script:
        "../src/OpenDengue/STGNN/tune.py"


rule train_stgnn:
    input:
        best_params = f"results/STGNN/{_name}/best_params.json",
    output:
        checkpoint = f"results/STGNN/{_name}/checkpoint.pt",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/train.py"


rule eval_stgnn:
    input:
        checkpoint = f"results/STGNN/{_name}/checkpoint.pt",
    output:
        metrics = f"results/STGNN/{_name}/metrics.json",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/eval.py"
