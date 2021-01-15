import mlflow
import shap
import pickle
import numpy as np
import sklearn
from mlflow.utils.environment import _mlflow_conda_env


def get_sklearn_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    pip_deps = ["shap=={}".format(shap.__version__)]

    return _mlflow_conda_env(
        additional_conda_deps=["scikit-learn={}".format(sklearn.__version__)],
        additional_pip_deps=pip_deps,
        additional_conda_channels=None,
    )

def test_sklearn_log_explainer():
    """
    Tests mlflow.shap log_explainer with mlflow serialization of the underlying model
    """

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
    
        explainer_original = shap.Explainer(model.predict, X, algorithm='permutation')
        shap_values_original = explainer_original(X[:5])

        mlflow.shap.log_explainer(explainer_original, "test_explainer")

        explainer_new = mlflow.shap.load_explainer("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_new(X[:5])

        assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
        assert type(explainer_original) == type(explainer_new)
        assert type(explainer_original.masker) == type(explainer_new.masker)

def test_sklearn_log_explainer_self_serialization():
    """
    Tests mlflow.shap log_explainer with SHAP internal serialization of the underlying model
    """

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
    
        explainer_original = shap.Explainer(model.predict, X, algorithm='permutation')
        shap_values_original = explainer_original(X[:5])

        mlflow.shap.log_explainer(explainer_original, "test_explainer", serialize_model_using_mlflow=False)

        explainer_new = mlflow.shap.load_explainer("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_new(X[:5])

        assert np.array_equal(shap_values_original.base_values,shap_values_new.base_values)
        assert type(explainer_original) == type(explainer_new)
        assert type(explainer_original.masker) == type(explainer_new.masker)

def test_sklearn_log_explainer_pyfunc():
    """
    Tests mlflow.shap log_explainer with mlflow serialization of the underlying model using pyfunc flavor
    """

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
    
        explainer_original = shap.Explainer(model.predict, X, algorithm='permutation')
        shap_values_original = explainer_original(X[:2])

        mlflow.shap.log_explainer(explainer_original, "test_explainer")

        explainer_pyfunc = mlflow.pyfunc.load_model("runs:/" + run_id + "/test_explainer")
        shap_values_new = explainer_pyfunc.predict(X[:2])
        

        assert np.array_equal(shap_values_original.base_values, shap_values_new['base_values'])
        assert shap_values_original.values.shape == shap_values_new['values'].shape

def test_pytorch_log_explainer():
    """
    Tests mlflow.shap log_explainer with mlflow serialization of the underlying pytorch model
    """

    import shap
    import mlflow
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es").cuda()

        # define the input sentences we want to translate
        data = [
            "In this picture, there are four persons: my father, my mother, my brother and my sister.",
            "Transformers have rapidly become the model of choice for NLP problems, replacing older recurrent neural network models"
        ]

        explainer_original = shap.Explainer(model, tokenizer)
        shap_values_original = explainer_original(data)
        mlflow.shap.log_explainer(explainer_original, artifact_path='shap_explainer')

        explainer_new = mlflow.shap.load_explainer("runs:/" + run_id + "/shap_explainer")
        shap_values_new = explainer_new(data)

        assert np.array_equal(shap_values_original[0].base_values, shap_values_new[0].base_values)

def test_pytorch_log_explainer_pyfunc():
    """
    Tests mlflow.shap log_explainer with mlflow serialization of the underlying model using pyfunc flavor
    """

    import shap
    import mlflow
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import pandas as pd

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-es").cuda()

        # define the input sentences we want to translate
        data = [
            "In this picture, there are four persons: my father, my mother, my brother and my sister.",
            "Transformers have rapidly become the model of choice for NLP problems, replacing older recurrent neural network models"
        ]

        explainer_original = shap.Explainer(model, tokenizer)
        shap_values_original = explainer_original(data)
        mlflow.shap.log_explainer(explainer_original, artifact_path='shap_explainer')

        explainer_pyfunc = mlflow.pyfunc.load_model("runs:/" + run_id + "/shap_explainer")
        shap_values_new = explainer_pyfunc.predict(pd.DataFrame(data=data)[0])

        assert np.array_equal(shap_values_original[0].base_values, shap_values_new['base_values'][0])
