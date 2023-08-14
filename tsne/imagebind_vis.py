from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import torchvision.datasets as datasets
from pycocotools.coco import COCO
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import open_clip
from open_clip import tokenizer
import dash
from dash import dcc, html, Dash

datapath = "/home/azureuser/val2017"
annpath = "/home/azureuser/annotations/instances_val2017.json"

coco = COCO(annpath)

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

templates_subset= [
    "A photo of a {}.",
    "A bad photo of the {}.",
    "A origami {}.",
    "A photo of the large {}.",
    "A {} in a video game.",
    "Art of the {}.",
    "A photo of the small {}."
]

text_list = []
for class_name in classes:
    class_templates = []
    #Change between all templates and subset templates
    for template in imagenet_templates:
        formatted_text = template.format(class_name)
        class_templates.append(formatted_text)
    text_list.append(class_templates)
# image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
# audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

with torch.no_grad():
    embeddings_all_text = []
    for class_templates in text_list:
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(class_templates, device),
        }
        embeddings = model(inputs)
        embeddings['text']/= torch.norm(embeddings['text'], dim=-1, keepdim=True)
        # embeddings['text'] *= 0.5
        embeddings['text'] = embeddings['text'].mean(dim=0)
        # embeddings['text'] /= embeddings['text'].norm()
        embeddings_all_text.append(embeddings['text'])

embeddings_all_text = torch.stack(embeddings_all_text, dim = 0)
tsne = TSNE(n_components=2).fit_transform(embeddings_all_text.cpu().numpy())

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
                    'text': classes,
                    'textposition': 'bottom center',
                    'type': 'scatter',
                    'marker': {'size': 8}
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
