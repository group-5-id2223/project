import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel
from nn_factory import nn_factory
import hopsworks

class BERT_classifier(nn.Module):
    def __init__(self, bertmodel, num_score):
        super(BERT_classifier, self).__init__()
        self.bertmodel = bertmodel
        self.dropout = nn.Dropout(p=bertmodel.config.hidden_dropout_prob)
        self.linear = nn.Linear(bertmodel.config.hidden_size, num_score)

    def forward(self, wrapped_input):
        hidden = self.bertmodel(**wrapped_input)
        last_hidden_state, pooler_output = hidden[0], hidden[1]
        output_value = self.linear(pooler_output).squeeze()
        score = torch.sigmoid(output_value) * 1000
        return score
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
bert = BertModel.from_pretrained("bert-base-uncased")

model = BERT_classifier(bert, 1)
model.load_state_dict(torch.load('inference_pipeline/model_1.pt', map_location=torch.device('cpu')))
model.eval()

project = hopsworks.login(project="id2223_enric")
fs = project.get_feature_store()
hackernews_fg = fs.get_feature_group(name="hackernews_fg", version=2)
query = hackernews_fg.select_all()
feature_view = fs.get_or_create_feature_view(name="hackernews_fv",
                                             version=2,
                                             query=query)
batch_data = feature_view.get_batch_data()

titles = batch_data['title'].tolist()

nn_obj = nn_factory(model, 'cpu', tokenizer)

for title in titles:
    print(nn_obj.predict(title))