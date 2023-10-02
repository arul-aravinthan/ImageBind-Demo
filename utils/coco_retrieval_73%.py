
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

# table['caption'] = table['caption'].apply(lambda x: x[0])

images_data = table['image'].tolist()
captions_data = table['caption'].tolist()
combined_data = list(zip(images_data, captions_data))


dataloader = DataLoader(combined_data, batch_size=1, shuffle=False)
# loop through all images and captions
for i, (images, texts) in enumerate(tqdm(dataloader)):
    processed_results = []
    # print(texts)
    image_inputs = load_and_transform_vision_data(images, 'cuda:0')
    text_inputs = data.load_and_transform_text(texts, 'cuda:0')
    inputs = {
        ModalityType.VISION: image_inputs,
        ModalityType.TEXT: text_inputs
    }
    # print(inputs[ModalityType.VISION].shape, inputs[ModalityType.TEXT].shape)
    with torch.no_grad():
        embeddings = model(inputs)

    caption_ids = []
    for caption_num in range(embeddings[ModalityType.TEXT].shape[0]):
        caption_ids.append(i)
    caption_results = {
            'image_embeds': embeddings[ModalityType.VISION],
            'text_embeds': embeddings[ModalityType.TEXT],
            'caption_ids': caption_ids,
            'image_ids': [i],
        }
    # pdb.set_trace()
    # print(caption_results['image_embeds'].shape, caption_results['text_embeds'].shape, caption_results['caption_ids'], caption_results['image_ids'])
    results = {'caption': caption_results}
    processed_results.append(results)

    evaluator.process(None, processed_results)

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
print(evaluator.evaluate())