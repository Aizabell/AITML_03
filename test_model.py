# test_model.py

import numpy as np
import mlflow.pyfunc

def load_model():
    # Use the same tracking URI as your app
    import mlflow
    mlflow.set_tracking_uri("https://admin:password@mlflow.ml.brain.cs.ait.ac.th")
    return mlflow.pyfunc.load_model("models:/ST125064-a3-model/Staging")

def test_model_input_shape():
    model = load_model()
    # Example: your model expects an input shape (n_samples, n_features), e.g., (1, 6)
    sample_input = np.array([[2015, 80, 1500, 1, 0, 0]])
    # Should not raise an error and return some prediction
    result = model.predict(sample_input)
    assert result is not None, "Model did not produce any output"

def test_model_output_shape():
    model = load_model()
    sample_input = np.array([[2015, 80, 1500, 1, 0, 0]])
    result = model.predict(sample_input)
    # For classification, expect a scalar or an array of shape (1,)
    result = np.array(result)
    assert result.shape[0] == 1, f"Expected output shape (1,), got {result.shape}"
