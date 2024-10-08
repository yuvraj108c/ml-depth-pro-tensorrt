import time
import tensorrt
import os
from trt_utilities import Engine

os.makedirs("checkpoints", exist_ok=True)
print("Tensorrt version: ", tensorrt.__version__)


def export_trt(trt_path=None, onnx_path=None, use_fp16=True):
    if trt_path is None:
        trt_path = input(
            "Enter the path to save the TensorRT engine (e.g ./realesrgan.engine): ")
    if onnx_path is None:
        onnx_path = input(
            "Enter the path to the ONNX model (e.g ./realesrgan.onnx): ")

    engine = Engine(trt_path)

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")
    print(f"Tensorrt engine saved at: {trt_path}")

    return ret


export_trt(trt_path="./checkpoints/depth_pro.engine",
           onnx_path="./checkpoints/depth_pro.onnx", use_fp16=True)
