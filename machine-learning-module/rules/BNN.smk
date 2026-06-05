
_name = config.get("name", "")


rule preprocess_BNN:
    input:
        data = config.get("data_path", ""),
    output:
        done = f"data/processed/BNN/{_name}/.preprocess.done",
    shell:
        """
        PYTHONPATH=src/OpenDengue/preprocess \
            python src/OpenDengue/preprocess/pipeline.py \
            --config {workflow.configfiles[-1]}
        touch {output.done}
        """


rule tune_BNN:
    input:
        done = f"data/processed/STGNN/{_name}/.preprocess.done",
    output:
        best_params = f"results/STGNN/{_name}/best_params.json",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/tune.py"


rule train_stgnn:
    input:
        done        = f"data/processed/STGNN/{_name}/.preprocess.done",
        best_params = f"results/STGNN/{_name}/best_params.json",
    output:
        checkpoint = f"results/STGNN/{_name}/checkpoint.pt",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/train.py"


rule eval_stgnn:
    input:
        done       = f"data/processed/STGNN/{_name}/.preprocess.done",
        checkpoint = f"results/STGNN/{_name}/checkpoint.pt",
    output:
        metrics = f"results/STGNN/{_name}/metrics.json",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/STGNN/eval.py"
