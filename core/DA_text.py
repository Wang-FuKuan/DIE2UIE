import torch.nn.functional as F
import torch

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def clip_preprocess_tensor(x):
    # x: [B,3,H,W], 0~1
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
    mean = x.new_tensor(CLIP_MEAN).view(1,3,1,1)
    std  = x.new_tensor(CLIP_STD).view(1,3,1,1)
    return (x - mean) / std

@torch.no_grad()
def perception(batch_img, text_features, clip_model):
    """
    batch_img: [B,3,H,W] 0~1
    text_features: [4,512]
    return:
      probs: [B,4]
      weighted_txt: [B,512]
    """
    x = clip_preprocess_tensor(batch_img)
    img_feat = clip_model.encode_image(x) 
    img_feat = F.normalize(img_feat, dim=-1)


    scale = getattr(clip_model, 'logit_scale', None)
    scale = scale.exp() if scale is not None else batch_img.new_tensor(100.0)

    logits = scale * (img_feat @ text_features.t())
    probs  = logits.softmax(dim=-1)
    weighted_txt = probs @ text_features
    return probs, weighted_txt