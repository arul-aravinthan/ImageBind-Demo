
import sys
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)
import pyarrow as pa
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from utils.xdecoder_retrieval_evaluation import RetrievalEvaluator
import io
import numpy as np
import os 
import pdb

datapath = "/home/azureuser/data/coco_caption_karpathy_test.arrow"

# extract image/text from arrow file
with pa.ipc.open_file(datapath) as reader:
    table = reader.read_all()

table = table.to_pydict()

evaluator = RetrievalEvaluator(distributed=False)
evaluator.reset()
# Model setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

#Preproccessing for images (Altered from Imagebind preprocessing, to work with JPEG)
def load_and_transform_vision_data(images, device):
    if images is None:
        return None

    image_outputs = []
    for image in images:
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
        image = Image.open(io.BytesIO(image)).convert("RGB") 
        image = data_transform(image).to(device)
        image_outputs.append(image)
    return torch.stack(image_outputs, dim=0)


images_data = table['image']
captions_data = table['caption']
combined_data = list(zip(images_data, captions_data))

def custom_collate_fn(batch):
    all_images = []
    all_captions = []
    image_ids = []
    for image, captions in batch:
        all_images.append(image)
        all_captions.append(data.load_and_transform_text(captions, 'cuda:0'))
    all_images = load_and_transform_vision_data(all_images, device)
    return all_images, all_captions

batch_size = 4
dataloader = DataLoader(combined_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# loop through all images and captions
i = 0
for images, texts in tqdm(dataloader):
    
    inputs = {
        ModalityType.VISION: images,
    }

    with torch.no_grad():
        embeddings = model(inputs)
    image_embeddings = embeddings[ModalityType.VISION]    

    for idx in range(len(texts)):
        processed_results = []
        inputs = {
            ModalityType.TEXT: texts[idx],
        }
        with torch.no_grad():
            embeddings = model(inputs)
        caption_ids = [i] * texts[idx].shape[0]
        caption_results = {
                'image_embeds': image_embeddings[idx].unsqueeze(0),
                'text_embeds': embeddings[ModalityType.TEXT],
                'caption_ids': caption_ids,
                'image_ids': [i],
            }
        results = {'caption': caption_results}
        processed_results.append(results)

        evaluator.process(None, processed_results)
        i += 1

del model, dataloader, table, combined_data, images_data, captions_data
torch.cuda.empty_cache()

print(evaluator.evaluate())