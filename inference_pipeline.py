import re
import torch
import torch.nn.functional as F

RUN_ON_MODAL=True
USE_GPU=True
BAYESIAN_SEARCH=True
BAYESIAN_ITERATIONS=7

from nbs.feature_processing import to_embedding

def predict_score(title: str, url: str) -> int:
    title_embedding = to_embedding([title]).unsqueeze(0)
    url_embedding = to_embedding([url]).unsqueeze(0)
    
    embeddings = torch.cat([title_embedding, url_embedding], dim=1)
    embeddings = F.softmax(embeddings, dim=-1)
    
    model = torch.load('nbs/model.pth')
    # call model from hopsworks?
    output = model(embeddings)
    scores = output * 280
    return int(scores)
    
    
def g():
    import re
    import transformers
    import hopsworks
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
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
    
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.ll1 = nn.Linear(768, 1024)
            self.bn1 = nn.BatchNorm1d(2)
            self.elu1 = nn.ELU()
            self.ll2 = nn.Linear(1024, 512)
            self.bn2 = nn.BatchNorm1d(2)
            self.elu2 = nn.ELU()
            self.llf = nn.Linear(512, 1)
            
        def forward(self, x):
            x = self.elu1(self.bn1(self.ll1(x)))
            x = self.elu2(self.bn2(self.ll2(x)))
            x = torch.sum(x, dim=1)
            x = self.llf(x)
            return x
    
    def extract_words_from_link(link):
    # Match alphanumeric sequences
        url_str = ""
        words = re.findall(r'\b\w+\b', link)
        remove_list = ['https', 'http', 'www']
        final_words = [w for w in words if not(w in remove_list)]
        for w in final_words:
            url_str += w + " "
        return url_str

    project = hopsworks.login(project='id2223_enric')
    fs = project.get_feature_store()
    
    hackernews_fg = fs.get_feature_group("hackernews_fg", 1)
    query = hackernews_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="hackernews_fv",
                                    version=1,
                                    description="Hackernews feature view",
                                    labels=["score"],
                                    query=query)
    #complete loading the df
    # we load the dataframe from hopsworks
    df = pass
    df_last = df.iloc[-10:]
    # df_last = #enric gets it
    
    title = df_last['title'].tolist()
    url = [extract_words_from_link(val) for val in df_last['url']]
    
    title_embedding = to_embedding(title).unsqueeze(1)
    url_embedding = to_embedding(url).unsqueeze(1)
    embeddings = torch.cat([title_embedding, url_embedding], dim=1)
    embeddings = F.softmax(embeddings, dim=-1)
    
    # model = load model from hopsworks after you upload it
    model = torch.load('nbs/model.pth')
    # call model from hopsworks?
    output = model(embeddings)
    scores = output * 280
    return scores
    
    
import modal
stub = modal.Stub()
image = modal.Image.debian_slim().pip_install(["hopsworks", "torch", "re", "transformers"])
@stub.function(image=image,
               schedule=modal.Period(days=7),
               secret=modal.Secret.from_name("hopsworks_pred"),
               timeout=60*60, # 60min timeout
               gpu="any",
               retries=1
               )
def f():
    g()

if __name__ == "__main__":
    if RUN_ON_MODAL:
        with stub.run():
            f.call()
    else:
        # NOTE: Create an .env file in the root directory if you want to run this locally
        g()
        
        
        