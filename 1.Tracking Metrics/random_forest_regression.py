import mlflow
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


mlflow.set_experiment("Diabetes Prediction")


def get_data():
    df = load_diabetes(as_frame=True)
    y = df.target
    X = df.data
    return X, y


with mlflow.start_run(run_name='RandomForest Regression') as run:
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    params = {
        "n_estimators": 50,
        "min_samples_split": 3
    }
    mlflow.log_params(params)
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.score(X_test, y_test)

    mlflow.log_metric('R2', pred)
