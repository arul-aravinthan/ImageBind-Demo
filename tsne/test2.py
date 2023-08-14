import torch
import torchvision.transforms as transforms
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from PIL import Image

# Define your COCO dataset paths
image_root = "/home/azureuser/val2017"  # Path to COCO images directory
annotation_file = "/home/azureuser/annotations/captions_val2017.json"

# Load COCO dataset
coco_dataset = CocoCaptions(root=image_root, annFile=annotation_file, transform=transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the model's input size
    transforms.ToTensor(),
]))

# Create a DataLoader for COCO dataset
batch_size = 3
coco_dataloader = DataLoader(coco_dataset, batch_size=batch_size, shuffle=True)

# Instantiate model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load and process data using the DataLoader
with torch.no_grad():
    for images, captions in coco_dataloader:
        images = images.to(device)
        captions = [caption[0] for caption in captions]  # Extract the first caption from each sample
        
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(captions, device),
            ModalityType.VISION: data.load_and_transform_vision_data(images, device)
        }

        embeddings = model(inputs)

        print(
            "Vision x Text: ",
            embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T,
        )
