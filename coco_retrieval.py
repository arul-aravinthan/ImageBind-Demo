
import pyarrow as pa
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
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

# extract image/ text fro arrow file
with pa.ipc.open_file(datapath) as reader:
    table = reader.read_all()

table = pd.DataFrame(table.to_pydict())
# for i in range(len(table['caption'])):
#     if(len(table['caption'][i]) > 5):
#         print(i, len(table['caption'][i]), table['caption'][i])
# define evaluator
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

# table['caption'] = table['caption'].apply(lambda x: list(x))

images_data = table['image'].tolist()
captions_data = table['caption'].tolist()
combined_data = list(zip(images_data, captions_data))

# for i in range(len(images_data)):
#     image_captions = {'image': images_data[i], 'captions': [captions_data[i]]}
#     combined_data.append(image_captions)

def custom_collate_fn(batch):
    all_images = []
    all_captions = []
    image_ids = []
    for image, captions in batch:
        # print(type(image), type(captions))
        all_images.append(image)
        all_captions.append(data.load_and_transform_text(captions, 'cuda:0'))
    all_images = load_and_transform_vision_data(all_images, device)
    return all_images, all_captions

batch_size = 4
dataloader = DataLoader(combined_data, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
# loop through all images and captions
i = 0
for images, texts in tqdm(dataloader):
    
    # print(curr_image_shape, curr_text_shape)
    inputs = {
        ModalityType.VISION: images,
    }
    # print(inputs[ModalityType.VISION].shape, inputs[ModalityType.TEXT].shape)

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
        # pdb.set_trace()
        results = {'caption': caption_results}
        processed_results.append(results)

        evaluator.process(None, processed_results)
        i += 1
    # if i > 100:
    #     break

# print(len(evaluator._image_embeds), len(evaluator._text_embeds))
# print(evaluator._image_embeds[0].shape, evaluator._text_embeds[0].shape)
# image_embeds = torch.cat(evaluator._image_embeds)
# text_embeds = torch.cat(evaluator._text_embeds)
# # print(image_embeds.shape, text_embeds.shape)
# image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
# text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
# print(image_embeds.shape, text_embeds.shape)
# scores = image_embeds @ text_embeds.t()
# # print(scores)
# print(scores.shape)
# print(scores.topk(10, dim=1))
# tiids = torch.tensor(evaluator._text_ids).view(-1).cuda()
# iids = torch.tensor(evaluator._image_ids).view(-1).cuda()
# print(tiids.shape)
# print(iids.shape)
# topk10 = scores.topk(10, dim=1)
# print(topk10.indices)
# print(tiids[topk10.indices])
del model, dataloader, table, combined_data, images_data, captions_data
torch.cuda.empty_cache()
print(evaluator.evaluate())