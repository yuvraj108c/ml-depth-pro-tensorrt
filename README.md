<div align="center">

# ML-Depth-Pro TensorRT ⚡

[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.5.0-green)](https://developer.nvidia.com/tensorrt)

</div>

This project provides a [TensorRT](https://github.com/NVIDIA/TensorRT) implementation of [ml-depth-pro](https://github.com/apple/ml-depth-pro) by Apple, enabling 5-6x faster inference speeds.

If you like the project, please give me a star! ⭐

---

## ⏱️ Performance

_Note: Inference was done in FP16, with a warm-up period of 5 frames. The reported time corresponds to the last inference._
| Device | Model Resolution| Inference Time (ms) |
| :----: | :-: | :-: |
|  H100  | 1536 x 1536  | 63 |

## 🚀 Installation

```bash
git clone https://github.com/apple/ml-depth-pro
cd ./ml-depth-pro
pip install -e .
bash get_pretrained_models.sh

git clone https://github.com/yuvraj108c/ml-depth-pro-tensorrt
mv ./ml-depth-pro-tensorrt/* . && rm -r ./ml-depth-pro-tensorrt
pip install -r requirements.txt
```

## 🛠️ Building onnx 
```bash
python export_onnx.py 
```

## 🛠️ Building tensorrt 
```bash
python export_trt.py 
```

## ⚡ Inference on single image
```bash
python infer_trt.py 
```

## 📓 Notes

The model currently supports images with a fixed resolution of 1536 x 1536  

## 🤖 Environment tested

Ubuntu 22.04 LTS, Cuda 12.4, Tensorrt 10.5.0, Python 3.10, H100 GPU
