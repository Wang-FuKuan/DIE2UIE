import torch
import torch.nn.functional as F
import clip_common
import clip
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToPILImage
from core.ram.models.ram_lora import ram

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip_common.load_clip_model(device='cpu')
model.to(device)
for para in model.parameters():
    para.requires_grad = False
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
    def __init__(self, initials=None):
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


prompt_pretrain_dir = '/models/best_prompt_round.pth'
prompts = ["a color-cast underwater image", "a hazy underwater image",
               "a non-uniform-illumination underwater image", "a visually good underwater image"]
learn_prompt = Prompts(prompts).to(device)
learn_prompt = torch.nn.DataParallel(learn_prompt)
learn_prompt.load_state_dict(torch.load(prompt_pretrain_dir, map_location=device))
text_encoder = TextEncoder(model)

embedding_prompt = learn_prompt.module.embedding_prompt
embedding_prompt.requires_grad = False
tokenized_prompts = torch.cat([tokenize_text(p) for p in [" ".join(["X"] * length_prompt)]])
text_features = text_encoder(embedding_prompt, tokenized_prompts)


RAM = ram(pretrained='/models/ram_swin_large_14m.pth',
              pretrained_condition = None,
              image_size=384,
              vit='swin_l')
RAM.eval().to(device, dtype=torch.float16)

tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def process_batch(batch_ram, ram_model, clip_model, clip_tokenizer, device):

    batch_prompts = []

    for i in range(batch_ram.size(0)):

        single_image = batch_ram[i]
        to_pil = ToPILImage()
        single_image = to_pil(single_image)

        single_image = tensor_transforms(single_image).unsqueeze(0).to(device)
        single_image = ram_transforms(single_image)
        single_image = single_image.half()
        with torch.no_grad():
            tags, _ = ram_model.generate_tag(single_image)
        validation_prompt = f"{tags[0]}"
        batch_prompts.append(validation_prompt)
    clip_model.eval()
    with torch.no_grad():
        tokens = clip.tokenize(batch_prompts).to(device)
        text_features = clip_model.encode_text(tokens)

    return text_features