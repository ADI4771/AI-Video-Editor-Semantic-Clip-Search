import torch
import clip
from PIL import Image

_model = None
_preprocess = None
_device = None


def load_model():
    """Load CLIP model once and cache it."""
    global _model, _preprocess, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _preprocess = clip.load("ViT-B/32", device=_device)
        _model.eval()
    return _model, _preprocess, _device


def get_image_embedding(pil_image: Image.Image) -> torch.Tensor:
    """Return normalized CLIP embedding for a PIL image."""
    model, preprocess, device = load_model()
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu()


def get_text_embedding(text: str) -> torch.Tensor:
    """Return normalized CLIP embedding for a text string."""
    model, _, device = load_model()
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu()
