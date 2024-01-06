import torch
import torch.nn.functional as F
from einops import rearrange

from tqdm import tqdm
import numpy as np
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hf_hub_model_name = "princeton-nlp/sup-simcse-bert-base-uncased"
hf_hub_model = None

def load_encoding_model():
    global hf_hub_model
    if hf_hub_model is None:
        from transformers import AutoModel, AutoTokenizer
        hf_hub_model = {
            "tokenizer": AutoTokenizer.from_pretrained(hf_hub_model_name),
            "model": AutoModel.from_pretrained(hf_hub_model_name)
        }
    return hf_hub_model

@torch.no_grad()
def to_embedding(data):
    hf_hub_model = load_encoding_model()
    inputs = hf_hub_model['tokenizer'](data, padding=True, truncation=True, return_tensors="pt")
    embedding = hf_hub_model['model'](**inputs, output_hidden_states=True, return_dict=True).pooler_output
    return embedding

def load_text_encoder(df):
    """
    Returns scores based on the title and url of the post
    """
    def encode(datalist):
        step = 1000
        outputs = []
        for i in tqdm(range(0, len(datalist), step)):
            embeddings = to_embedding(datalist[i:i+1])
            embeddings = F.softmax(embeddings, dim=1)
            outputs.append(embeddings.cpu())
            
        embeddings = torch.stack(outputs, dim=0)
        embeddings = rearrange(embeddings, 'n s dim -> (n s) dim')
        
    title_list = df['title'].tolist()
    url_list = df['url'].tolist()
    
    title_embeddings = encode(title_list)
    url_embeddings = encode(url_list)
    
    return title_embeddings, url_embeddings
            
        
        