from contextlib import contextmanager
import os
import tempfile
import pickle
import yaml
import shap

import numpy as np
import sklearn
import torch

import mlflow
import types
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.uri import append_to_uri_path
from mlflow.models import Model

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "shap"

_MAXIMUM_BACKGROUND_DATA_SIZE = 100
_DEFAULT_ARTIFACT_PATH = "model_explanations_shap"
_SUMMARY_BAR_PLOT_FILE_NAME = "summary_bar_plot.png"
_BASE_VALUES_FILE_NAME = "base_values.npy"
_SHAP_VALUES_FILE_NAME = "shap_values.npy"
_UNKNOWN_MODEL_FLAVOR = "unknown"
_UNDERLYING_MODEL_SUBPATH = 'underlying_model'

def get_underlying_model_flavor(model):
    """
    Find the underlying models flavor.

    :param model: underlying model of the explainer.
    """

    unwrapped_model = model.model
    if isinstance(unwrapped_model, types.FunctionType):
        return 'python_function'
    elif isinstance(unwrapped_model, types.MethodType):
        model_object = unwrapped_model.__self__
        if issubclass(type(model_object), sklearn.base.BaseEstimator):
            return 'sklearn'
    elif issubclass(type(unwrapped_model), torch.nn.Module):
        return 'pytorch'
        # TODO: Add checks for other types of mlflow models
    return _UNKNOWN_MODEL_FLAVOR

def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    pip_deps = ["shap=={}".format(shap.__version__)]

    return _mlflow_conda_env(
        additional_conda_deps=[],
        additional_pip_deps=pip_deps,
        additional_conda_channels=None,
    )

def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return _SHAPWrapper(path)

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
def log_explainer(
    explainer,
    artifact_path,
    serialize_model_using_mlflow=True,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """
    Log an SHAP explainer as an MLflow artifact for the current run.

    :param explainer: SHAP explainer to be saved.
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
                                'shap=0.37.0'
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
        explainer=explainer,
        conda_env=conda_env,
        serialize_model_using_mlflow=serialize_model_using_mlflow,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
    )


@experimental
def save_model(
    explainer,
    path,
    serialize_model_using_mlflow = True,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
):
    """
    Save a SHAP explainer to a path on the local file system. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.shap`
        - :py:mod:`mlflow.pyfunc`

    :param explainer: SHAP explainer to be saved.
    :param path: Local path where the explainer is to be saved.
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
                                'shap=0.37.0'
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
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
    """

    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path), error_code=RESOURCE_ALREADY_EXISTS
        )

    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    
    
    underlying_model_flavor = None
    underlying_model_path = None

    # saving the underlying model if required
    if serialize_model_using_mlflow == True:
        underlying_model_flavor = get_underlying_model_flavor(explainer.model)

        if underlying_model_flavor != _UNKNOWN_MODEL_FLAVOR:
            explainer.model.save = None
            underlying_model_path = os.path.join(path, _UNDERLYING_MODEL_SUBPATH)
        
        if underlying_model_flavor == 'sklearn':
            mlflow.sklearn.save_model(explainer.model.model.__self__, underlying_model_path)
        elif underlying_model_flavor == 'pytorch':
            mlflow.pytorch.save_model(explainer.model.model, underlying_model_path)
    
    # saving the explainer object
    explainer_data_subpath = "explainer.shap"
    explainer_output_path = os.path.join(path, explainer_data_subpath)
    explainer_output_file_handle = open(explainer_output_path, "wb")
    explainer.save(explainer_output_file_handle)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)

    # merging the conda environment generated by serializing the underlying model
    if underlying_model_path is not None:
        underlying_model_conda_path = os.path.join(underlying_model_path, 'conda.yaml')
        underlying_model_conda_file = open(underlying_model_conda_path, 'r')
        underlying_model_conda_env = yaml.safe_load(underlying_model_conda_file)
        conda_env = _merge_environments(conda_env, underlying_model_conda_env)
    
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.shap",
        model_path=explainer_data_subpath,
        underlying_model_flavor=underlying_model_flavor,
        env=conda_env_subpath
    )

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        shap_version=shap.__version__,
        serialized_explainer=explainer_data_subpath,
        underlying_model_flavor=underlying_model_flavor
    )

    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

def _merge_environments(shap_environment, model_environment):
    """
    Merge conda environments of underlying model and shap.

    :param shap_environment: SHAP conda environment.
    :param model_environment: Underlying model conda environment.
    """
    
    merged_conda_channels = list(set(shap_environment['channels'] + model_environment['channels']))
    merged_conda_deps = set()
    merged_pip_deps = set()

    for dependency in shap_environment['dependencies']:
        if isinstance(dependency, dict) and dependency['pip']:
            for pip_dependency in dependency['pip']:
                if pip_dependency != 'mlflow':
                    merged_pip_deps.add(pip_dependency)
        else:
            if dependency.split('=')[0] != 'python' and dependency.split('=')[0] != 'pip':
                merged_conda_deps.add(dependency)

    for dependency in model_environment['dependencies']:
        if isinstance(dependency, dict) and dependency['pip']:
            for pip_dependency in dependency['pip']:
                if pip_dependency != 'mlflow':
                    merged_pip_deps.add(pip_dependency)
        else :
            if dependency.split('=')[0] != 'python' and dependency.split('=')[0] != 'pip':
                merged_conda_deps.add(dependency)

    merged_conda_deps = list(merged_conda_deps)
    merged_pip_deps = list(merged_pip_deps)
    
    return _mlflow_conda_env(
        additional_conda_deps=merged_conda_deps,
        additional_pip_deps=merged_pip_deps,
        additional_conda_channels=merged_conda_channels
    )

@experimental
def load_explainer(model_uri):
    """
    Load a SHAP explainer from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :return: A SHAP explainer.
    """

    local_explainer_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_explainer_path, flavor_name=FLAVOR_NAME)
    shap_explainer_artifacts_path = os.path.join(local_explainer_path, flavor_conf["serialized_explainer"])
    underlying_model_flavor = flavor_conf["underlying_model_flavor"]
    model = None
    
    if underlying_model_flavor != _UNKNOWN_MODEL_FLAVOR:
        underlying_model_path = os.path.join(local_explainer_path, _UNDERLYING_MODEL_SUBPATH)
        if underlying_model_flavor == 'sklearn':
            model = mlflow.sklearn._load_pyfunc(underlying_model_path).predict
        elif underlying_model_flavor == 'pytorch':
            model = mlflow.pytorch._load_model(os.path.join(underlying_model_path, 'data'))
    
    return _load_explainer(explainer_file=shap_explainer_artifacts_path, model=model)


@experimental
def _load_explainer(explainer_file, model = None):
    """
    Load a SHAP explainer saved as an MLflow artifact on the local file system.

    :param explainer_file: Local filesystem path to the MLflow Model saved with the ``shap`` flavor
    :param model: model to override underlying explainer model.
    """
    def inject_model_loader(in_file):
        pickle.load(in_file) # No-Op to move file pointer forward
        return model

    with open(explainer_file, "rb") as explainer:
        model_loader = None
        if model is not None:
            model_loader = inject_model_loader
        explainer = shap.Explainer.load(explainer, model_loader=model_loader)
        return explainer

class _SHAPWrapper:
    def __init__(self, path):
        flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
        shap_explainer_artifacts_path = os.path.join(path, flavor_conf["serialized_explainer"])
        underlying_model_flavor = flavor_conf["underlying_model_flavor"]
        model = None
        if underlying_model_flavor != _UNKNOWN_MODEL_FLAVOR:
            underlying_model_path = os.path.join(path, _UNDERLYING_MODEL_SUBPATH)
            if underlying_model_flavor == 'sklearn':
                model = mlflow.sklearn._load_pyfunc(underlying_model_path).predict
            elif underlying_model_flavor == 'pytorch':
                model = mlflow.pytorch._load_model(os.path.join(underlying_model_path, 'data'))

        self.explainer = _load_explainer(explainer_file=shap_explainer_artifacts_path, model=model)

    def predict(self, dataframe):
        explanation_serialized = {}
        shap_values = self.explainer(dataframe.values)
        explanation_serialized['values'] = shap_values.values.tolist()
        if hasattr(shap_values, 'base_values'):
            explanation_serialized['base_values'] = shap_values.base_values.tolist()
        return explanation_serialized