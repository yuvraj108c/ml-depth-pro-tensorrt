import torch
from trt_utilities import Engine
from polygraphy import cuda
import time
from PIL import Image
import numpy as np
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

IMG_SIZE = 1536
warmup = 5
device = "cuda"
precision = torch.float16
f_px = None

engine_path = "./checkpoints/depth_pro.engine"
image_path = "./data/example.jpg"
output_path = "./data/result.png"

transform = Compose(
    [
        ToTensor(),
        Lambda(lambda x: x.to(device)),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ConvertImageDtype(precision),
    ]
)

# Load and preprocess an image.
image = Image.open(image_path).convert('RGB')
image_t = transform(image).unsqueeze(0)

# Setup tensorrt engine
engine = Engine(engine_path)
engine.load()
engine.activate()
engine.allocate_buffers()
    
_, _, H, W = image_t.shape
resize = H != IMG_SIZE or W != IMG_SIZE
interpolation_mode = "bilinear"

if resize:
    image_t = torch.nn.functional.interpolate(
        image_t,
        size=(IMG_SIZE, IMG_SIZE),
        mode=interpolation_mode,
        align_corners=False,
    )

# warm up
print(f"Warming up {warmup} times")
for i in range(warmup):
    s = time.time()
    stream = cuda.Stream()
    prediction= engine.infer({"input": image_t}, stream, use_cuda_graph=False)
    stream.synchronize()
    e = time.time()
    print("Inference time: ", (e-s)*1000, "ms")

s = time.time()
stream = cuda.Stream()
prediction= engine.infer({"input": image_t}, stream, use_cuda_graph=False)
stream.synchronize()
e = time.time()
print("Final inference time: ", (e-s)*1000, "ms")

canonical_inverse_depth, fov_deg = prediction["depth"], prediction["focallength_px"]

if f_px is None:
    f_px = 0.5 * W / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
        
inverse_depth = canonical_inverse_depth * (W / f_px)
f_px = f_px.squeeze()
print("Focal length in px:", f_px.item())

if resize:
    inverse_depth = torch.nn.functional.interpolate(
        inverse_depth, size=(H, W), mode=interpolation_mode, align_corners=False
    )

depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
depth = depth.squeeze().cpu().numpy()

# Normalize depth values
depth_min, depth_max = np.percentile(depth, [2, 98])  # Use percentiles to remove outliers
depth_normalized = np.clip((depth - depth_min) / (depth_max - depth_min), 0, 1)

# Invert the depth map so closer objects are lighter
depth_normalized = 1 - depth_normalized    

# Convert to 8-bit grayscale
depth_gray = (depth_normalized * 255).astype(np.uint8)

# Create and save the image
depth_image = Image.fromarray(depth_gray, mode='L')
depth_image.save(output_path, format="PNG")
