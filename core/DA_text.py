import torch.nn.functional as F
import torch
import clip
import torch.nn as nn

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

length_prompt = 16
def tokenize_text(text):
    tokenized_text = clip.tokenize(text)
    return tokenized_text

class TextEncoder(nn.Module):
    def __init__(self, clip_model):

        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class Prompts(nn.Module):
    def __init__(self, initials=None, model=None):
        super(Prompts, self).__init__()
        self.text_encoder = TextEncoder(model)
        text = tokenize_text(initials).cuda()
        self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()

    def forward(self, tensor, flag=1):

        tokenized_prompts = torch.cat([tokenize_text(p) for p in [" ".join(["X"] * length_prompt)]])
        text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts)

        probs = []
        class_outputs = []
        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            nor = torch.norm(text_features, dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ (text_features / nor).T)
            similarity2 = similarity.softmax(dim=-1)
            class_output = torch.argmax(similarity2, dim=-1)

            if i == 0:
                probs = similarity
                class_outputs = class_output
            else:
                probs = torch.cat([probs, similarity], dim=0)
                class_outputs = torch.cat([class_outputs, class_output], dim=0)

        return probs, class_outputs


def clip_preprocess_tensor(x):
    # x: [B,3,H,W], 0~1
    if x.shape[-2:] != (224, 224):
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
    mean = x.new_tensor(CLIP_MEAN).view(1,3,1,1)
    std  = x.new_tensor(CLIP_STD).view(1,3,1,1)
    return (x - mean) / std

@torch.no_grad()
def perception(batch_img, clip_model, device):
    """
    batch_img: [B,3,H,W] 0~1
    text_features: [4,512]
    return:
      probs: [B,4]
      weighted_txt: [B,512]
    """
    model = clip_model
    prompt_pretrain_dir = '/models/best_prompt_round.pth'
    prompts = ["a color-cast underwater image", "a hazy underwater image",
                "a non-uniform-illumination underwater image", "a visually good underwater image"]
    learn_prompt = Prompts(prompts, model).to(device)
    learn_prompt = torch.nn.DataParallel(learn_prompt)
    learn_prompt.load_state_dict(torch.load(prompt_pretrain_dir, map_location=device))
    text_encoder = TextEncoder(model)

    embedding_prompt = learn_prompt.module.embedding_prompt
    embedding_prompt.requires_grad = False
    tokenized_prompts = torch.cat([tokenize_text(p) for p in [" ".join(["X"] * length_prompt)]])
    text_features = text_encoder(embedding_prompt, tokenized_prompts)

    x = clip_preprocess_tensor(batch_img)
    img_feat = clip_model.encode_image(x) 
    img_feat = F.normalize(img_feat, dim=-1)

    scale = getattr(clip_model, 'logit_scale', None)
    scale = scale.exp() if scale is not None else batch_img.new_tensor(100.0)

    logits = scale * (img_feat @ text_features.t())
    probs  = logits.softmax(dim=-1)
    weighted_txt = probs @ text_features
    return probs, weighted_txt
