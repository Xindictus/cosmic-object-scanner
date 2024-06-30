# import torch

# torch.backends.cuda.enabled = True
# torch.backends.cuda.cufft_plan_cache.max_size = 2**10  # Limit CUDA cache size

from ultralytics import YOLO

# Load a model
# load a pretrained model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(data="data.yaml", epochs=2, imgsz=608)
