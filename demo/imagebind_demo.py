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
import pickle as pkl
import gradio as gr

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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

def compute_similarity(textbox, image, audio):
    inputs = {}
    truthList = []
    if len(textbox) != 0:
        inputs[ModalityType.TEXT] = data.load_and_transform_text([textbox], device)
    if image is not None:
        inputs[ModalityType.VISION] = data.load_and_transform_vision_data([image], device)
    if audio is not None:
        inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data([audio], device)
    with torch.no_grad():
        embeddings = model(inputs)
    similarities = []
    for modality in embeddings.keys():
        embeddings[modality] /= torch.norm(embeddings[modality], dim=-1, keepdim=True)
        similarities.append(embeddings[modality] @ image_embeddings.T)
    similarity = torch.mean(torch.stack(similarities), dim=0)
    index_results = torch.topk(similarity, 30, dim=1)    
    path_results = [image_paths[index] for _, index in enumerate(index_results.indices[0])]
    return path_results

customCSS = """
.grid-container.svelte-1b19cri.svelte-1b19cri {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}

.thumbnail-lg.svelte-1b19cri.svelte-1b19cri{
    width: unset;
    height: 35vh;
    aspect-ratio: auto;
}
"""

with gr.Blocks(theme='gradio/monochrome', css=customCSS) as demo:
    gr.Markdown("## Enter a combination of image, text, and audio input!")
    with gr.Row():
        imageInput = gr.Image(type='filepath')
        textboxInput = gr.Textbox(placeholder="Enter text here...")
        audioInput = gr.Audio(type='filepath', label="Audio")
    galleryOutput = gr.Gallery(label="Gallery")
    btn = gr.Button("Search")
    btn.click(fn = compute_similarity, inputs = [textboxInput, imageInput, audioInput], outputs = galleryOutput)
    
demo.launch(share=True)
