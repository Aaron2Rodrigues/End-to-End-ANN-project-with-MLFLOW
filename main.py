import os
import keras
import pandas as pd  
import numpy as np
import mlflow 
from mlflow.models import infer_signature
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Importing the dataset
url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"
data = pd.read_csv(url, sep=';')
data.head()

# Splitting the dataset into training and testing sets
X = data.drop("quality", axis=1).values
y = data[["quality"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Validation data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

signature = infer_signature(X_train, y_train)

def train_model(params, epochs, X_train, X_test, X_val, y_train, y_test, y_val):
    mean = np.mean(X_train, axis=0)
    var = np.var(X_train, axis=0)

    model = keras.models.Sequential([
        keras.Input(shape=(X_train.shape[1],)),
        keras.layers.Normalization(mean=mean, variance=var),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Compiling the model
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=params['lr'], momentum=params['momentum']),
                  loss='mean_squared_error',
                  metrics=[keras.metrics.RootMeanSquaredError()])

    # Training the model with params and tracking with MLFlow
    with mlflow.start_run(nested=True):
        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32)

        # Evaluating the model
        eval_result = model.evaluate(X_val, y_val, batch_size=32)
        eval_rmse = eval_result[1]

        # Logging the metrics
        mlflow.log_params(params)
        mlflow.log_metric("rmse", eval_rmse)

        # Logging the model
        mlflow.tensorflow.log_model(model, "ANN-model", signature=signature)

        return {"loss": eval_rmse, "status": STATUS_OK, "model": model}

# Defining the objective function
def objective(params):
    # MLFlow will track the params and metrics for each set of params
    result = train_model(params, epochs=3, X_train=X_train, X_test=X_test, X_val=X_val, 
                         y_train=y_train, y_test=y_test, y_val=y_val)
    return result

# Hyperparameter space
space = {
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-1)),
    "momentum": hp.uniform("momentum", 0.1, 0.9)
}

# Setting MLFlow experiment
mlflow.set_experiment("Wine-Quality")

# Start MLFlow run and conduct hyperparameter tuning with Hyperopt
with mlflow.start_run():
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=3, trials=trials)
    
    best_run = sorted(trials.results, key=lambda x: x['loss'])[0]

    # Logging the best params, loss, and model
    mlflow.log_params(best)
    mlflow.log_metric("eval_rmse", best_run['loss'])
    mlflow.tensorflow.log_model(best_run['model'], "ANN-model", signature=signature)

# Print out the best params
print(f"Best params: {best}")
print(f"Best Loss: {best_run['loss']}")
