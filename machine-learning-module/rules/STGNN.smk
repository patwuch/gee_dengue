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
        checkpoint = f"results/STGNN/{_name}/best_model.pt",
        losses     = f"results/STGNN/{_name}/train_val_losses.json",
        loss_curve = f"results/STGNN/{_name}/loss_curves.png",
    params:
        cfg         = lambda wc: workflow.configfiles[-1],
        best_params = f"results/STGNN/{_name}/best_params.json",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/train.py"


rule test_stgnn:
    input:
        model       = f"results/STGNN/{_name}/best_model.pt",
        best_params = f"results/STGNN/{_name}/best_params.json",
    output:
        predictions = f"results/STGNN/{_name}/test_predictions.npz",
        metrics     = f"results/STGNN/{_name}/metrics.json",
        pred_plot   = f"results/STGNN/{_name}/predictions.png",
        scatter     = f"results/STGNN/{_name}/scatter.png",
    params:
        cfg         = lambda wc: workflow.configfiles[-1],
        best_params = f"results/STGNN/{_name}/best_params.json",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/test.py"


# ---------------------------------------------------------------------------
# Production training rule — only active when best_params_source is set.
#
# Trains on the full dataset (no test holdout) using hyperparameters already
# found by the baseline sweep, so no new tuning run is required.
#
# Invoke:
#   snakemake results/STGNN/production_logIR/best_model.pt \
#       --configfile config/OpenDengue/stgnn_logIR_production.yaml \
#       --cores 4 --resources gpu=1
# ---------------------------------------------------------------------------

if config.get("best_params_source"):
    _params_source = config["best_params_source"]

    rule train_stgnn_production:
        input:
            best_params = f"results/STGNN/{_params_source}/best_params.json",
            tensors     = expand(
                "data/processed/STGNN/{name}/window_{w}/tensors.pt",
                name = _name,
                w    = _window_sizes,
            ),
            scaler      = f"data/processed/STGNN/{_name}/scaler.pkl",
        output:
            checkpoint = f"results/STGNN/{_name}/best_model.pt",
            losses     = f"results/STGNN/{_name}/train_val_losses.json",
            loss_curve = f"results/STGNN/{_name}/loss_curves.png",
        params:
            cfg         = lambda wc: workflow.configfiles[-1],
            best_params = f"results/STGNN/{_params_source}/best_params.json",
        resources:
            gpu = 1,
        script:
            "../src/OpenDengue/STGNN/train.py"
