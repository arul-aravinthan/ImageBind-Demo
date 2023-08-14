import torch
import numpy as np
from PIL import Image
import open_clip
from sklearn.manifold import TSNE
import dash
from dash import dcc
from dash import html


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')
model.eval()
model.to(device)

# Define the list of text strings (classes)
classes = [
    "none",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush",
]

# Encode each text string and concatenate them to create the matrix
with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = []
    for text in classes:
        text_input = tokenizer([text]).cuda()
        text_embedding = model.encode_text(text_input)
        text_features.append(text_embedding)
    for i in range(len(text_features)):
        text_features[i] = text_features[i].cpu().numpy()
    text_features = np.concatenate(text_features, axis=0)
    text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)

# Visualize the matrix using t-SNE
tsne = TSNE(n_components=2).fit_transform(text_features)

tsne_x_scaled = (tsne[:, 0] - tsne[:, 0].min()) / (tsne[:, 0].max() - tsne[:, 0].min())
tsne_y_scaled = (tsne[:, 1] - tsne[:, 1].min()) / (tsne[:, 1].max() - tsne[:, 1].min())

# Create Dash app to display the visualization
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("OPEN_CLIP - COCO Classes Visualization"),
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

if __name__ == '__main__':
    app.run_server(debug=True)
