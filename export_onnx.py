import depth_pro
import torch
import os

DEVICE = "cuda"
model, transform = depth_pro.create_model_and_transforms(device=DEVICE)
model.eval()

x = torch.randn(1, 3, 1536, 1536).to(DEVICE)

os.makedirs("checkpoints", exist_ok=True)
onnx_save_path = os.path.join("checkpoints", "depth_pro.onnx")

with torch.no_grad():
    torch.onnx.export(model,
                      x,
                      onnx_save_path,
                      opset_version=19,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['depth', "focallength_px"],
                      )
    print(f"ONNX model exported to: {onnx_save_path}")