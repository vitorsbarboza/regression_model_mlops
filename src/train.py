import os
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dotenv import load_dotenv
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment(os.getenv('MLFLOW_EXPERIMENT_NAME'))

def main():
    df = pd.read_csv('data/processed.csv')
    X = df.drop('meses_ate_desligamento', axis=1)
    y = df['meses_ate_desligamento']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mlflow.log_metric('mae', mae)
        mlflow.sklearn.log_model(model, 'model')

        # Exportar para ONNX
        initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open("model/model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        mlflow.log_artifact("model/model.onnx")

if __name__ == '__main__':
    main()
