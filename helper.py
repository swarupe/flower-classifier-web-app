import io

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

def load_model():
    checkpoint = 'flower_classifier2.pt'
    model_state = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    model = models.densenet121(pretrained=True)
    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'], strict=False)
    model.class_to_idx = model_state['class_to_idx']
    model.eval()
    return model

def img_tensor(image_bytes):
    img_transforms = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    img = Image.open(io.BytesIO(image_bytes))
    return img_transforms(img).unsqueeze(0)