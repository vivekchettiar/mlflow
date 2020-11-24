from contextlib import contextmanager
import os
import tempfile

import numpy as np

import mlflow
from mlflow.utils.annotations import experimental
from mlflow.utils.uri import append_to_uri_path
from mlflow.models import Model

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "shap"

_MAXIMUM_BACKGROUND_DATA_SIZE = 100
_DEFAULT_ARTIFACT_PATH = "model_explanations_shap"
_SUMMARY_BAR_PLOT_FILE_NAME = "summary_bar_plot.png"
_BASE_VALUES_FILE_NAME = "base_values.npy"
_SHAP_VALUES_FILE_NAME = "shap_values.npy"


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return path

@contextmanager
def _log_artifact_contextmanager(out_file, artifact_path=None):
    """
    A context manager to make it easier to log an artifact.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, out_file)
        yield tmp_path
        mlflow.log_artifact(tmp_path, artifact_path)


def _log_numpy(numpy_obj, out_file, artifact_path=None):
    """
    Log a numpy object.
    """
    with _log_artifact_contextmanager(out_file, artifact_path) as tmp_path:
        np.save(tmp_path, numpy_obj)


def _log_matplotlib_figure(fig, out_file, artifact_path=None):
    """
    Log a matplotlib figure.
    """
    with _log_artifact_contextmanager(out_file, artifact_path) as tmp_path:
        fig.savefig(tmp_path)


@experimental
def log_explanation(predict_function, features, artifact_path=None):
    r"""
    Given a ``predict_function`` capable of computing ML model output on the provided ``features``,
    computes and logs explanations of an ML model's output. Explanations are logged as a directory
    of artifacts containing the following items generated by `SHAP`_ (SHapley Additive
    exPlanations).

        - Base values
        - SHAP values (computed using `shap.KernelExplainer`_)
        - Summary bar plot (shows the average impact of each feature on model output)

    :param predict_function:
        A function to compute the output of a model (e.g. ``predict_proba`` method of
        scikit-learn classifiers). Must have the following signature:

        .. code-block:: python

            def predict_function(X) -> pred:
                ...

        - ``X``: An array-like object whose shape should be (# samples, # features).
        - ``pred``: An array-like object whose shape should be (# samples) for
          a regressor or (# classes, # samples) for a classifier. For a classifier,
          the values in ``pred`` should correspond to the predicted probability of each class.

        Acceptable array-like object types:

            - ``numpy.array``
            - ``pandas.DataFrame``
            - ``shap.common.DenseData``
            - ``scipy.sparse matrix``

    :param features:
        A matrix of features to compute SHAP values with. The provided features should
        have shape (# samples, # features), and can be either of the array-like object
        types listed above.

        .. note::
            Background data for `shap.KernelExplainer`_ is generated by subsampling ``features``
            with `shap.kmeans`_. The background data size is limited to 100 rows for performance
            reasons.

    :param artifact_path:
        The run-relative artifact path to which the explanation is saved.
        If unspecified, defaults to "model_explanations_shap".

    :return: Artifact URI of the logged explanations.

    .. _SHAP: https://github.com/slundberg/shap

    .. _shap.KernelExplainer: https://shap.readthedocs.io/en/latest/generated
        /shap.KernelExplainer.html#shap.KernelExplainer

    .. _shap.kmeans: https://github.com/slundberg/shap/blob/v0.36.0/shap/utils/_legacy.py#L9

    .. code-block:: python
        :caption: Example

        import os

        import numpy as np
        import pandas as pd
        from sklearn.datasets import load_boston
        from sklearn.linear_model import LinearRegression

        import mlflow

        # prepare training data
        dataset = load_boston()
        X = pd.DataFrame(dataset.data[:50, :8], columns=dataset.feature_names[:8])
        y = dataset.target[:50]

        # train a model
        model = LinearRegression()
        model.fit(X, y)

        # log an explanation
        with mlflow.start_run() as run:
            mlflow.shap.log_explanation(model.predict, X)

        # list artifacts
        client = mlflow.tracking.MlflowClient()
        artifact_path = "model_explanations_shap"
        artifacts = [x.path for x in client.list_artifacts(run.info.run_id, artifact_path)]
        print("# artifacts:")
        print(artifacts)

        # load back the logged explanation
        dst_path = client.download_artifacts(run.info.run_id, artifact_path)
        base_values = np.load(os.path.join(dst_path, "base_values.npy"))
        shap_values = np.load(os.path.join(dst_path, "shap_values.npy"))

        print("\n# base_values:")
        print(base_values)
        print("\n# shap_values:")
        print(shap_values[:3])

    .. code-block:: text
        :caption: Output

        # artifacts:
        ['model_explanations_shap/base_values.npy',
         'model_explanations_shap/shap_values.npy',
         'model_explanations_shap/summary_bar_plot.png']

        # base_values:
        20.502000000000002

        # shap_values:
        [[ 2.09975523  0.4746513   7.63759026  0.        ]
         [ 2.00883109 -0.18816665 -0.14419184  0.        ]
         [ 2.00891772 -0.18816665 -0.14419184  0.        ]]

    .. figure:: ../_static/images/shap-ui-screenshot.png

        Logged artifacts
    """
    import matplotlib.pyplot as plt
    import shap

    artifact_path = _DEFAULT_ARTIFACT_PATH if artifact_path is None else artifact_path
    background_data = shap.kmeans(features, min(_MAXIMUM_BACKGROUND_DATA_SIZE, len(features)))
    explainer = shap.KernelExplainer(predict_function, background_data)
    shap_values = explainer.shap_values(features)

    _log_numpy(explainer.expected_value, _BASE_VALUES_FILE_NAME, artifact_path)
    _log_numpy(shap_values, _SHAP_VALUES_FILE_NAME, artifact_path)

    shap.summary_plot(shap_values, features, plot_type="bar", show=False)
    fig = plt.gcf()
    fig.tight_layout()
    _log_matplotlib_figure(fig, _SUMMARY_BAR_PLOT_FILE_NAME, artifact_path)
    plt.close(fig)

    return append_to_uri_path(mlflow.active_run().info.artifact_uri, artifact_path)



@experimental
def log_model(
    shap_explainer,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    # signature: ModelSignature = None,
    # input_example: ModelInputExample = None,
    # await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """
    Log an ONNX model as an MLflow artifact for the current run.

    :param onnx_model: ONNX model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.6.0',
                                'onnx=1.4.1',
                                'onnxruntime=0.3.0'
                            ]
                        }
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.shap,
        shap_explainer=shap_explainer,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        # signature=signature,
        # input_example=input_example,
        # await_registration_for=await_registration_for,
    )


@experimental
def save_model(
    shap_explainer,
    path,
    conda_env=None,
    mlflow_model=None,
    # signature: ModelSignature = None,
    # input_example: ModelInputExample = None,
):
    """
    """
    import os

    if not os.path.exists(path):
        os.mkdir(path)
        with open(os.path.join(path,"temp"), "w") as stream:
            stream.write("i am here")


    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        sklearn_version=sklearn.__version__,
        serialization_format=serialization_format,
    )

    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    print("Called save_model")


@experimental
def load_model(model_uri):
    """
    Load an ONNX model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see the
                      `Artifacts Documentation <https://www.mlflow.org/docs/latest/
                      tracking.html#artifact-stores>`_.

    :return: An ONNX model instance.

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return _load_model(model_file=onnx_model_artifacts_path)
