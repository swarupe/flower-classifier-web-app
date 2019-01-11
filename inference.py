import json

from helper import img_tensor, load_model

with open('cat_to_name.json') as f:
    cat_to_name = json.load(f)

with open('class_to_idx.json') as f:
    class_to_idx = json.load(f)

idx_to_class = {v:k for k,v in class_to_idx.items()}

model = load_model()

def predict_flower(image_bytes):
    imgTensor = img_tensor(image_bytes)
    outputs = model.forward(imgTensor)
    _, pred = outputs.max(1)
    category = pred.item()
    class_idx = idx_to_class[category]
    flower_name = cat_to_name[class_idx]
    return flower_name
    

