"""
demo/ports.py — the real frozen heterogeneous models as measure-valued PORTS. This is a boundary module
(like geometry/): it turns model forward passes into feature vectors; nothing downstream sees coordinates
except through geometry/clouds (which normalizes to (D, w)). All models frozen.

Ports (R2/R3):
  vit        — google/vit-base-patch16-224           (vision)
  mobilenet  — torchvision mobilenet_v3_small          (vision, different family)
  minilm     — all-MiniLM-L6-v2                         (text)
  clip       — openai/clip-vit-base-patch32            (CHANNEL-NATIVE: vision tower + text tower)
CLIP's two towers are the natural cross-modal channel pair (R2). Cross-modal is satisfied: text (minilm,
clip-text) + two vision families (vit, mobilenet, clip-image).
"""
import warnings, functools
warnings.filterwarnings("ignore")
import numpy as np
import torch

torch.set_num_threads(4)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _prep_images(images, size=224):
    import torch.nn.functional as Fn
    arr = np.stack([np.asarray(im.convert("RGB").resize((size, size)), np.float32) / 255 for im in images])
    x = torch.tensor(arr.transpose(0, 3, 1, 2))
    return (x - _MEAN) / _STD


class Port:
    def __init__(self, name, modality, channels):
        self.name = name
        self.modality = modality      # 'vision' | 'text'
        self.channels = channels      # list of channel names (R2)

    def encode(self, inputs, channel=0):
        raise NotImplementedError


class ViTPort(Port):
    def __init__(self):
        super().__init__("vit", "vision", ["cls", "patch-mean"])
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained("google/vit-base-patch16-224").eval()

    @torch.no_grad()
    def encode(self, images, channel=0):
        h = self.m(pixel_values=_prep_images(images)).last_hidden_state   # (N,197,768)
        return (h[:, 0] if channel == 0 else h[:, 1:].mean(1)).numpy()


class MobileNetPort(Port):
    def __init__(self):
        super().__init__("mobilenet", "vision", ["penultimate", "logits"])
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        self.m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).eval()

    @torch.no_grad()
    def encode(self, images, channel=0):
        x = _prep_images(images)
        feat = self.m.features(x)
        pooled = self.m.avgpool(feat).flatten(1)          # penultimate
        if channel == 0:
            return pooled.numpy()
        return self.m.classifier(pooled).numpy()           # logits


class MiniLMPort(Port):
    def __init__(self):
        super().__init__("minilm", "text", ["mean", "cls"])
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()

    @torch.no_grad()
    def encode(self, texts, channel=0):
        e = self.tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=64)
        h = self.m(**e).last_hidden_state
        if channel == 1:
            return h[:, 0].numpy()                         # CLS
        mask = e["attention_mask"].unsqueeze(-1).float()
        return ((h * mask).sum(1) / mask.sum(1)).numpy()   # mean-pool


class CLIPPort(Port):
    """Channel-native: one model, two towers. modality='both'; encode routes by `channel` (0=vision,1=text)."""
    def __init__(self):
        super().__init__("clip", "both", ["vision-tower", "text-tower"])
        from transformers import CLIPModel, CLIPProcessor
        self.m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    @torch.no_grad()
    def encode(self, inputs, channel=0):
        if channel == 0:                                   # vision tower
            px = self.proc(images=[im.convert("RGB") for im in inputs], return_tensors="pt")["pixel_values"]
            pooled = self.m.vision_model(pixel_values=px).pooler_output
            return self.m.visual_projection(pooled).numpy()
        e = self.proc(text=list(inputs), return_tensors="pt", padding=True, truncation=True)  # text tower
        pooled = self.m.text_model(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).pooler_output
        return self.m.text_projection(pooled).numpy()


class SigLIPPort(Port):
    """Channel-native second vision<->text dual-encoder bridge (independently trained from CLIP).
    modality='both'; encode routes by `channel` (0=vision tower, 1=text tower). Mirrors CLIPPort exactly so
    it drops into the same member roster; the point is a SECOND, independently-trained cross-modal overlap."""
    def __init__(self):
        super().__init__("siglip", "both", ["vision-tower", "text-tower"])
        from transformers import AutoModel, AutoProcessor
        self.m = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval()
        self.proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

    @staticmethod
    def _feat(out):
        # transformers 5.x get_*_features returns a pooled-output object; older returns a tensor.
        return (out.pooler_output if hasattr(out, "pooler_output") else out).numpy()

    @torch.no_grad()
    def encode(self, inputs, channel=0):
        if channel == 0:                                   # vision tower
            px = self.proc(images=[im.convert("RGB") for im in inputs], return_tensors="pt")["pixel_values"]
            return self._feat(self.m.get_image_features(pixel_values=px))
        e = self.proc(text=list(inputs), return_tensors="pt", padding="max_length", truncation=True)  # text
        return self._feat(self.m.get_text_features(input_ids=e["input_ids"]))


@functools.lru_cache(maxsize=1)
def load_ports():
    """All frozen ports, loaded once. Returns dict name -> Port."""
    return {p.name: p for p in [ViTPort(), MobileNetPort(), MiniLMPort(), CLIPPort(), SigLIPPort()]}
