import onnxruntime as ort
import onnx
import numpy as np

model_path = "logs/rsl_rl/upkie_velocity/bests/default_no_push_tmp.onnx"

# Check the model
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# Example of input
observation = np.array([0.0] * 22, dtype=np.float32)
x = {"obs": [observation,]}

ort_sess = ort.InferenceSession(model_path)
outputs = ort_sess.run(None, x)

print("ONNX Runtime output:", outputs)
