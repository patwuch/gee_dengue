rule preprocess_indonesia_data:
    input:
        data = "data/indonesia_smk/processed_data.csv",
    output:
        model = "models/indonesia_smk/model.pkl",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/Indonesia_SMK/train_model.py"

rule tune_indonesia_xgboost:
    input:
        data = "data/indonesia_smk/processed_data.csv",
    output:
        params_csv = "reports/tables/indonesia_smk/xgboost_params.csv",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/Indonesia_SMK/tune_xgboost.py"

rule train_indonesia_xgboost:
    input:
        params_csv = "reports/tables/indonesia_smk/xgboost_params.csv",
        data       = "data/indonesia_smk/processed_data.csv",
    output:
        model = "models/indonesia_smk/xgboost_model.json",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/Indonesia_SMK/train_xgboost.py"

rule eval_indonesia_xgboost:
    input:
        model = "models/indonesia_smk/xgboost_model.json",
        data  = "data/indonesia_smk/processed_data.csv",
    output:
        results_csv = "reports/tables/indonesia_smk/xgboost_results.csv",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/Indonesia_SMK/eval_xgboost.py"

rule shap_analysis_indonesia_xgboost:
    input:
        model = "models/indonesia_smk/xgboost_model.json",
        data  = "data/indonesia_smk/processed_data.csv",
    output:
        shap_csv = "reports/tables/indonesia_smk/xgboost_shap.csv",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/Indonesia_SMK/shap_analysis.py"

rule shap_visualisation_indonesia_xgboost:
    input:
        shap_csv = "reports/tables/indonesia_smk/xgboost_shap.csv",
    output:
        shap_plot = "reports/figures/indonesia_smk/xgboost_shap.png",
    resources:
        gpu = 1,
    script:
        "../src/OpenDengue/Indonesia_SMK/visualise_shap.py"