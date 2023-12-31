import sys
pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from utils.xdecoder_retrieval_evaluation import RetrievalEvaluator
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import pdb

evaluator = RetrievalEvaluator(distributed=False)
evaluator.reset()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
rerun = True
datapath = "data/clotho_validation_evaluation"
captions_path = "data/clotho_captions_validation.csv"

# Model setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

df = pd.read_csv(captions_path)
audio_names = df['file_name'].tolist()

# DataLoader
# batch_size = 1
# dataloader = DataLoader(audio_names, batch_size=batch_size, shuffle=False)

# Initialize embeddings list
all_embeddings_text = []
all_embeddings_audio = []

j = 0
# Process batches
for i in tqdm(range(len(df))):
    #using only one caption 
    text_paths = [df['caption_1'][i], df['caption_2'][i], df['caption_3'][i], df['caption_4'][i], df['caption_5'][i]]
    audio_paths = [os.path.join(datapath , df['file_name'][i])]
    audio_files = data.load_and_transform_audio_data(audio_paths, device)
    text_files = data.load_and_transform_text(text_paths, device)
    inputs = {
        ModalityType.AUDIO: audio_files,
        ModalityType.TEXT: text_files,
    }
    
    with torch.no_grad():
        embeddings = model(inputs)
    # print(embeddings[ModalityType.AUDIO].shape, embeddings[ModalityType.TEXT].shape)
    processed_results = []
    caption_ids = [j] * embeddings[ModalityType.TEXT].shape[0]
    caption_results = {
            'image_embeds': embeddings[ModalityType.AUDIO],
            'text_embeds': embeddings[ModalityType.TEXT],
            'caption_ids': caption_ids,
            'image_ids': [j],
        }
    results = {'caption': caption_results}
    # print(results['caption']['image_embeds'].shape, results['caption']['text_embeds'].shape, results['caption']['caption_ids'], results['caption']['image_ids'])
    processed_results.append(results)
    evaluator.process(None, processed_results)
    j += 1
        # embeddings[ModalityType.AUDIO] /= torch.norm(embeddings[ModalityType.AUDIO], dim=-1, keepdim=True)
        # embeddings[ModalityType.TEXT] /= torch.norm(embeddings[ModalityType.TEXT], dim=-1, keepdim=True)
        # averaged_text = torch.mean(embeddings[ModalityType.TEXT], dim=0, keepdim=True)

        # all_embeddings_audio.append(embeddings[ModalityType.AUDIO])
        # all_embeddings_text.append(averaged_text)


    # Stack the embeddings across all batches
#     audio_embeddings = torch.cat(all_embeddings_audio, dim=0)
#     text_embeddings = torch.cat(all_embeddings_text, dim=0)

#     similarity = torch.matmul(text_embeddings, audio_embeddings.T)
#     similarity2 = torch.softmax(similarity, dim=-1)
#     pkl.dump(similarity2, open('/home/azureuser/ImageBind-Demo/similarity2.pkl', 'wb'))
#     pkl.dump(similarity, open('/home/azureuser/ImageBind-Demo/similarity.pkl', 'wb'))   
# else:
#     similarity = pkl.load(open('/home/azureuser/ImageBind-Demo/similarity.pkl', 'rb'))
#     similarity2 = pkl.load(open('/home/azureuser/ImageBind-Demo/similarity2.pkl', 'rb'))
# diagonal_elements = torch.diagonal(similarity)
# count = 0
# for i in range(len(similarity)):
#     if torch.argmax(similarity[i]) == i:
#         count += 1
# print(count, count/len(similarity))
print(evaluator.evaluate())
