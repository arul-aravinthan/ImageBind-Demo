from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]
video_paths=[".assets/sample-5s.mp4"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

t1 = data.load_and_transform_video_data(video_paths, device)

print(t1.shape)
# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# Load data
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
    
}
with torch.no_grad():
    embeddings = model(inputs)

print(embeddings[ModalityType.VISION].shape, embeddings[ModalityType.TEXT].shape)
print(
    "Vision x Text: ",
    embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T,
)
print(
    "Audio x Text: ",
    embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T,
)
print(
    "Vision x Audio: ",
    embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T,
)

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])
