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

_ML_PROCESSED = PROJECT_ROOT / "data" / "processed" / "machine-learning"


rule preprocess_stgnn:
    message:
        "Preprocessing for STGNN with experiment name '{_name}' and window sizes {_window_sizes}."
    input:
        str(MERGED_DENGUE_ENV_DATA),
    output:
        tensors = expand(
            str(_ML_PROCESSED / "STGNN/{name}/window_{w}/tensors.pt"),
            name=_name,
            w=_window_sizes,
        ),
        scaler  = str(_ML_PROCESSED / f"STGNN/{_name}/preprocessing_params.json"),
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    script:
        f"{workflow.basedir}/src/OpenDengue/STGNN/preprocess/pipeline.py"

rule tune_stgnn:
    input:
        tensors = expand(
            str(_ML_PROCESSED / "STGNN/{name}/window_{w}/tensors.pt"),
            name = _name,
            w    = _window_sizes,
        ),
        scaler  = str(_ML_PROCESSED / f"STGNN/{_name}/preprocessing_params.json"),
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


rule choropleth_stgnn:
    input:
        predictions = f"results/STGNN/{_name}/test_predictions.npz",
        best_params = f"results/STGNN/{_name}/best_params.json",
    output:
        actuals       = f"results/STGNN/{_name}/actual_ir_14months.png",
        pred_vs_actual = f"results/STGNN/{_name}/pred_vs_actual.png",
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    script:
        "../src/OpenDengue/STGNN/choropleth.py"


rule explain_attention_stgnn:
    message:
        "Visualising GAT attention weights for experiment '{_name}'."
    input:
        model       = f"results/STGNN/{_name}/best_model.pt",
        best_params = f"results/STGNN/{_name}/best_params.json",
    output:
        weights     = f"results/STGNN/{_name}/attention_weights.npz",
        graph       = f"results/STGNN/{_name}/attention_graph.png",
        over_time   = f"results/STGNN/{_name}/attention_over_time.png",
        interactive = f"results/STGNN/{_name}/attention_graph_interactive.html",
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    script:
        "../src/OpenDengue/STGNN/explain_attention.py"


rule explain_shap_stgnn:
    message:
        "Computing SHAP Expected Gradients feature attributions for experiment '{_name}'."
    input:
        model       = f"results/STGNN/{_name}/best_model.pt",
        best_params = f"results/STGNN/{_name}/best_params.json",
    output:
        attributions = f"results/STGNN/{_name}/shap_attributions.npz",
        feature_plot = f"results/STGNN/{_name}/shap_feature_importance.png",
        heatmap      = f"results/STGNN/{_name}/shap_node_time_heatmap.png",
    params:
        cfg = lambda wc: workflow.configfiles[-1],
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/explain_shap.py"


# ---------------------------------------------------------------------------
# Production training rule — only active when best_params_source is set.
# ---------------------------------------------------------------------------

if config.get("best_params_source"):
    _params_source = config["best_params_source"]

    rule train_stgnn_production:
        input:
            best_params = f"results/STGNN/{_params_source}/best_params.json",
            tensors     = expand(
                str(_ML_PROCESSED / "STGNN/{name}/window_{w}/tensors.pt"),
                name = _name,
                w    = _window_sizes,
            ),
            scaler      = str(_ML_PROCESSED / f"STGNN/{_name}/preprocessing_params.json"),
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
