import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pycocotools.coco import COCO
from PIL import Image
from imagebind import data
from imagebind.models.imagebind_model import ModalityType
from imagebind.models import imagebind_model
from tqdm import tqdm
from utils.xdecoder_retrieval_evaluation import RetrievalEvaluator
import pickle as pkl

# Paths
datapath = "/home/azureuser/data/val2017"
annpath = "/home/azureuser/data/annotations/instances_val2017.json"

# COCO Dataset
coco = COCO(annpath)
image_paths = [os.path.join(datapath, coco.loadImgs(img_id)[0]['file_name']) for img_id in coco.getImgIds()]

# Model setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

if not os.path.exists('/home/azureuser/ImageBind-Demo/image_embeddings.pkl'):
    # Transformation for images
    def load_and_transform_vision_data(image_paths, device):
        image_outputs = []
        for image_path in image_paths:
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
            with open(image_path, "rb") as fopen:
                image = Image.open(fopen).convert("RGB")

            image = data_transform(image).to(device)
            image_outputs.append(image)
        return torch.stack(image_outputs, dim=0)

    # DataLoader
    batch_size = 32
    dataloader = DataLoader(image_paths, batch_size=batch_size, shuffle=False)

    # Initialize embeddings list
    all_embeddings = []

    # Process batches
    for batch_paths in tqdm(dataloader, desc="Processing batches"):
        images = load_and_transform_vision_data(batch_paths, device)
        inputs = {
            ModalityType.VISION: images,
        }
        
        with torch.no_grad():
            embeddings = model(inputs)
        
        embeddings[ModalityType.VISION] /= torch.norm(embeddings[ModalityType.VISION], dim=-1, keepdim=True)

        all_embeddings.append(embeddings[ModalityType.VISION])

    # Stack the embeddings across all batches
    image_embeddings = torch.cat(all_embeddings, dim=0)

    # Save the embeddings
    with open('image_embeddings.pkl', 'wb') as f:
        pkl.dump(image_embeddings, f)
else:
    with open('image_embeddings.pkl', 'rb') as f:
        image_embeddings = pkl.load(f)

image_embeddings = image_embeddings.to(device)
all_captions = []
for image_id in coco.getImgIds():
    annotations =  coco.loadAnns(coco.getAnnIds(imgIds=image_id))
    captions = [annotation['caption'] for annotation in annotations]
    all_captions.append(captions)

dataloader = DataLoader(all_captions, batch_size=1, shuffle=False)
all_embeddings_text = []
for batch_captions in tqdm(dataloader, desc="Processing batches"):
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(batch_captions, device),
    }
    
    with torch.no_grad():
        embeddings = model(inputs)
    
    embeddings[ModalityType.TEXT] /= torch.norm(embeddings[ModalityType.TEXT], dim=-1, keepdim=True)
    averaged_text = torch.mean(embeddings[ModalityType.TEXT], dim=0, keepdim=True)

    all_embeddings_text.append(averaged_text)

text_embeddings = torch.cat(all_embeddings_text, dim=0)

similarity = torch.matmul(text_embeddings, image_embeddings.T)

pkl.dump(similarity, open('/home/azureuser/ImageBind-Demo/similarity.pkl', 'wb'))

diagonal_elements = torch.diagonal(similarity)
count = 0
for i in range(len(similarity)):
    if torch.max(similarity[i]) == diagonal_elements[i]:
        count += 1
print(count, count/len(similarity))