#Code modified from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
import sys
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import io
import numpy as np
import os 
from PIL import Image
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from utils import imagenet_templates
from utils import imagenet_classes

imagenet_templates = imagenet_templates.imagenet_templates
imagenet_classes = imagenet_classes.imagenet_classes

datapath = "data/ImageNet"

data_transform = transforms.Compose(
    [
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)

# Model setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = data.load_and_transform_text(texts, device) #tokenize
            inputs = {
                ModalityType.TEXT: texts,
            }
            with torch.no_grad():
                embeddings = model(inputs) #embed with text encoder
            class_embeddings = embeddings[ModalityType.TEXT]
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


zeroshot_weights = zeroshot_classifier(imagenet_classes, imagenet_templates)

batch_size = 32
imagenet_data = datasets.ImageNet(datapath, split="val", transform=data_transform)
loader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, num_workers=4)

# loop through all images and captions

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

top1, top5, top10, n = 0., 0., 0., 0.
for images, target in tqdm(loader):
    images = images.to(device)
    target = target.to(device)
    inputs = {
        ModalityType.VISION: images,
    }
    with torch.no_grad():
        embeddings = model(inputs)

    image_embeddings = embeddings[ModalityType.VISION]
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

    logits = 100. * image_embeddings @ zeroshot_weights
    acc1, acc5, acc10 = accuracy(logits, target, topk=(1, 5, 10))
    top1 += acc1
    top5 += acc5
    top10 += acc10

    n += images.size(0)   

top1 = (top1 / n) * 100
top5 = (top5 / n) * 100
top10 = (top10 / n) * 100

print(f"Top-1 accuracy: {top1:.2f}")
print(f"Top-5 accuracy: {top5:.2f}")
print(f"Top-10 accuracy: {top10:.2f}")
