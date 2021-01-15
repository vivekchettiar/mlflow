import mlflow
import shap
import sklearn

mlflow.start_run()
run = mlflow.active_run()
run_id = run.info.run_id

# prepare training data
X, y = shap.datasets.boston()

# train a model
model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# create an explainer
explainer_original = shap.Explainer(model.predict, X, algorithm='permutation',)

# log an explainer
mlflow.shap.log_explainer(explainer_original, artifact_path='shap_explainer')


# load back the explainer
explainer_new = mlflow.shap.load_explainer(f'runs:/{run_id}/shap_explainer')

# run explainer on data
shap_values = explainer_new(X[:5])

print(shap_values)

mlflow.end_run()