# clip_utils.py

import torchvision.transforms as transforms
import torch
import clip
import torch.nn as nn
from torch.nn import functional as F
import sys


device = "cuda" if torch.cuda.is_available() else "cpu"
#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="/CLIP_mine/clip_model/")#"ViT-B/32"
model.to(device)
for para in model.parameters():
	para.requires_grad = False


# Function to load the CLIP model and preprocess function
def load_clip_model(device='cpu'):
    model, preprocess = clip.load("ViT-B/32", device=torch.device(device), download_root='/CLIP_mine/clip_model/')
    return model, preprocess

# Function to load the CLIP model and preprocess function
def load_resnet_model(device='cpu'):
    model, preprocess = clip.load("RN101", device=torch.device(device), download_root='/CLIP_mine/clip_model/')
    return model, preprocess

# Function to tokenize text
def tokenize_text(text):
    tokenized_text = clip.tokenize(text)
    return tokenized_text


def get_clip_score(tensor, words):
    score = 0
    for i in range(tensor.shape[0]):
        # image preprocess
        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                               (0.26862954, 0.26130258, 0.27577711))
        img_resize = transforms.Resize((224, 224))
        image2 = img_resize(tensor[i])
        image = clip_normalizer(image2).unsqueeze(0)
        # get probabilitis
        text = clip.tokenize(words).to(device)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1)
        # 2-word-compared probability
        # prob = probs[0][0]/probs[0][1]#you may need to change this line for more words comparison
        prob = probs[0][0]
        score = score + prob

    return score


class L_clip(nn.Module):
    def __init__(self):
        super(L_clip, self).__init__()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, light):
        k1 = get_clip_score(x, ["dark", "normal light"])
        if light:
            k2 = get_clip_score(x, ["noisy photo", "clear photo"])
            return (k1 + k2) / 2
        return k1


class Prompts(nn.Module):
    def __init__(self, initials=None):
        super(Prompts, self).__init__()
        if initials != None:
            text = clip.tokenize(initials).cuda()
            with torch.no_grad():
                self.text_features = model.encode_text(text).cuda()
        else:
            self.text_features = torch.nn.init.xavier_normal_(nn.Parameter(torch.cuda.FloatTensor(2, 512))).cuda()

    def forward(self, tensor):
        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            nor = torch.norm(self.text_features, dim=-1, keepdim=True)
            similarity = (model.logit_scale.exp() * image_features @ (self.text_features / nor).T).softmax(dim=-1)
            if (i == 0):
                probs = similarity
            else:
                probs = torch.cat([probs, similarity], dim=0)
        return probs


learn_prompt = Prompts().cuda()
clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
img_resize = transforms.Resize((224, 224))


def get_clip_score_from_feature0(tensor, text_features):
    score = 0
    for i in range(tensor.shape[0]):
        image2 = img_resize(tensor[i])
        image = clip_normalizer(image2.reshape(1, 3, 224, 224))

        image_features = model.encode_image(image)
        image_nor = image_features.norm(dim=-1, keepdim=True)
        nor = text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * (image_features / image_nor) @ (text_features / nor).T).softmax(dim=-1)
        probs = similarity
        prob = probs[0][0]
        score = score + prob
    score = score / tensor.shape[0]
    return score


def get_clip_score_from_feature(tensor, text_features, normal_index=3):
    score_normal = 0
    score_degradation = 0
    for i in range(tensor.shape[0]):
        image2 = img_resize(tensor[i])
        image = clip_normalizer(image2.reshape(1, 3, 224, 224))

        # Get image features from the CLIP model
        image_features = model.encode_image(image)
        image_nor = image_features.norm(dim=-1, keepdim=True)
        nor = text_features.norm(dim=-1, keepdim=True)

        # Calculate the similarity with all text prompts
        similarity = (100.0 * (image_features / image_nor) @ (text_features / nor).T).softmax(dim=-1)
        probs = similarity

        # Use the probability that corresponds to "normal"
        prob_normal = probs[0][normal_index]
        score_normal += prob_normal

        # Calculate mean probability for other degradation prompts
        prob_degradation = (probs[0][:normal_index].sum() + probs[0][normal_index + 1:].sum()) / 3
        score_degradation += prob_degradation

        # Average over batch
    score_normal = score_normal / tensor.shape[0]
    score_degradation = score_degradation / tensor.shape[0]

    # Define the final loss: encourage high "normal" score and low "degradation" score
    return -(score_normal - score_degradation)

class L_clip_from_feature(nn.Module):
    def __init__(self):
        super(L_clip_from_feature, self).__init__()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, text_features):
        # k1 = get_clip_score_from_feature(x, text_features)
        k1 = get_clip_score_from_feature(x, text_features, normal_index=3)
        return k1