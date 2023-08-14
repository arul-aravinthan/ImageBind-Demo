import torch
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import dash
from dash import dcc, html, Dash
from PIL import Image
import pdb
from sklearn.manifold import TSNE
import os 
import pickle
from tqdm import tqdm
from random import sample
from collections import defaultdict

num_samples = 10000

datapath = "/home/azureuser/val2017"
annpath = "/home/azureuser/annotations/instances_val2017.json"

coco = COCO(annpath)
#Change amount of samples here
annotations = coco.loadAnns(sample(coco.getAnnIds(), num_samples))
# Get category IDs and names
category_ids = coco.getCatIds()
categories = coco.loadCats(category_ids)

# Extract category names
classes = [category['name'] for category in categories]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

if not os.path.exists('/home/azureuser/ImageBind/class_embeddings.pkl'):
    text_list = {}  
    for class_name in classes:
        class_templates = []
        for template in imagenet_templates:
            formatted_text = template.format(class_name)
            class_templates.append(formatted_text)
        text_list[class_name] = class_templates
    embeddings_classes = {} 
    #Getting text embeddings for each class
    with torch.no_grad():
        for class_name, class_templates in text_list.items(): 
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(class_templates, device),
            }
            embeddings = model(inputs)
            embeddings['text'] /= torch.norm(embeddings['text'], dim=-1, keepdim=True)
            # embeddings['text'] *= 0.5
            embeddings['text'] = embeddings['text'].mean(dim=0)
            # embeddings['text'] /= embeddings['text'].norm()
            embeddings_classes[class_name] = embeddings['text'] 

    # print(embeddings_classes)
    print("Done getting text embeddings")
    with open('/home/azureuser/ImageBind/class_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings_classes, f)
else:
    with open('/home/azureuser/ImageBind/class_embeddings.pkl', 'rb') as f:
        embeddings_classes = pickle.load(f)

# print(embeddings_classes)

# Getting image embeddings
image_embeddings_final = defaultdict(list)

# Apply resizing, cropping, and transformations
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

pbar = tqdm(total=len(annotations), desc="Processing Annotations")
with torch.no_grad():
    for annotation in annotations:
        image_id = annotation['image_id']
        image_path = coco.loadImgs(image_id)[0]['file_name']
        category = coco.loadCats(annotation['category_id'])[0]['name']
        # Load the image
        image = Image.open(datapath + "/" + image_path).convert("RGB")
        
        # Get the bounding box from the COCO annotation
        x, y, width, height = annotation['bbox']
        
        # Crop the image based on the bounding box
        cropped_image = image.crop((x, y, x + width, y + height))

        # Apply the defined transformations to the cropped image
        transformed_image = data_transform(cropped_image).to(device)
        # Pass the transformed image through the model to get image embeddings
        image_embedding = model({
            ModalityType.VISION: transformed_image.unsqueeze(0) 
        })[ModalityType.VISION]

        # Normalize the image embeddings
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        
        image_embeddings_final[category].append(image_embedding.squeeze(0))
        pbar.update(1)
pbar.close()

embeddings_sum = []
category_list = []
# Average the image embeddings for each class
for category, embeddings_list in image_embeddings_final.items():
    average_image_embedding = torch.stack(embeddings_list, dim = 0).mean(dim=0)
    embeddings_sum.append(torch.add(average_image_embedding, embeddings_classes[category]))
    category_list.append(category)

embeddings_sum = torch.stack(embeddings_sum, dim = 0)


tsne = TSNE(n_components=2).fit_transform(embeddings_sum.cpu().numpy())

tsne_x_scaled = (tsne[:, 0] - tsne[:, 0].min()) / (tsne[:, 0].max() - tsne[:, 0].min())
tsne_y_scaled = (tsne[:, 1] - tsne[:, 1].min()) / (tsne[:, 1].max() - tsne[:, 1].min())


app = Dash(__name__)
app.layout = html.Div([
    html.H1("IMAGEBIND - COCO Classes Visualization"),
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                {
                    'x': tsne_x_scaled,
                    'y': tsne_y_scaled,
                    'mode': 'markers+text',
                    'text': category_list,
                    'textposition': 'bottom center',
                    'hoverinfo': 'text',
                    'type': 'scatter',
                    'marker': {
                        'size': 8,
                        # 'color': list(range(len(classes))),
                        # 'colorscale': 'Viridis',
                        # 'showscale': True
                        }
                }
            ],
            'layout': {
                'title': 'COCO CLASSES',
                'xaxis': {'title': 'X'},
                'yaxis': {'title': 'Y'},
            }
        }
    )
])
if __name__ == "__main__":
    app.run_server(debug=False)



