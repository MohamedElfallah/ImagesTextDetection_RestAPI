from django import urls
import torch 
from PIL import Image
import io
from django.conf import settings
import os

# loading Yolo Model
def get_model(): 
    model = torch.hub.load("ultralytics/yolov5", 'custom', path=os.path.join(settings.BASE_DIR, "model/weights.pt"))
    model.conf = 0.5
    return model

# convert bytes to image
def bytes_to_image(image_bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
