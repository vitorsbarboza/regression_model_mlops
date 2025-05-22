import pandas as pd
import onnxruntime as rt
import numpy as np

def predict(input_data):
    sess = rt.InferenceSession('model/model.onnx')
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: input_data.astype(np.float32)})
    return pred_onx[0]

if __name__ == '__main__':
    # Exemplo de uso
    df = pd.read_csv('data/processed.csv')
    X = df.drop('meses_ate_desligamento', axis=1)
    print(predict(X.values))
