import modal
from nn_factory import nn_factory

LOCAL=False
if LOCAL == False:
    packages = ["hopsworks", "torch", "transformers", 
                "pandas", "huggingface_hub", "seaborn"]

    stub = modal.Stub('hackernews-score-pred-inference')
    image = modal.Image.debian_slim().pip_install(packages)

    @stub.function(image=image,
                schedule=modal.Period(days=1), 
                secret=modal.Secret.from_name("hopsworks.ai"),
                )
    def f():
        g()

def g():
    import torch
    import torch.nn as nn
    import hopsworks
    from transformers import BertTokenizer, BertModel
    from datetime import datetime, timedelta
    from huggingface_hub import hf_hub_download
    import pandas as pd
    import seaborn as sns

    class BERT_classifier(nn.Module):
        def __init__(self, bertmodel, num_score):
            super(BERT_classifier, self).__init__()
            self.bertmodel = bertmodel
            self.dropout = nn.Dropout(p=bertmodel.config.hidden_dropout_prob)
            self.linear = nn.Linear(bertmodel.config.hidden_size, num_score)

        def forward(self, wrapped_input):
            hidden = self.bertmodel(**wrapped_input)
            _, pooler_output = hidden[0], hidden[1]
            output_value = self.linear(pooler_output).squeeze()
            score = torch.sigmoid(output_value) * 1000
            return score

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained("bert-base-uncased")

    model_dir = hf_hub_download(repo_id="ID2223/hackernews_upvotes_predictor_model", filename="model_1.pt", repo_type="model")
    model = BERT_classifier(bert, 1)
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    model.eval()

    project = hopsworks.login(project='id2223_enric')
    fs = project.get_feature_store()
    
    hackernews_fg = fs.get_feature_group("hackernews_fg", 2)
    query = hackernews_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="hackernews_fv",
                                    version=2,
                                    description="Hackernews feature view",
                                    labels=["score"],
                                    query=query)

    batch_data = feature_view.get_batch_data(
        start_time=datetime.now() - timedelta(days=1),
        read_options={"use_hive": True}
    )
    df = hackernews_fg.read(read_options={"use_hive": True})
    new_df = batch_data.merge(df[['title', 'score']], on='title', how='left')
    new_df['predicted_score'] = pd.Series(dtype='int')
    nn_obj = nn_factory(model, 'cpu', tokenizer)
    

    for i in new_df.index:
        title = new_df['title'][i]
        predicted_score = nn_obj.predict(title)
        new_df.loc[i, 'predicted_score'] = predicted_score
    
    monitor_fg = fs.get_or_create_feature_group(name="hackernews_predictions",
                                            version=1,
                                            primary_key=["id"],
                                            description="Hackernews Predictions/Outcome Monitoring"
                                            )
    new_df = new_df.filter(['id', 'title', 'score', 'predicted_score'])
    monitor_fg.insert(new_df, overwrite=True)

    dataset_api = project.get_dataset_api() 

    sns.set_theme(style="darkgrid")
    sp = sns.scatterplot(x="score", y="predicted_score", data=new_df)
    sp.xaxis.set_label_text("Actual score")
    sp.yaxis.set_label_text("Predicted score")
    sp.set_title("Actual vs Predicted score")
    sp.set(xlim=(-10, 1000), ylim=(-10, 1000))
    fig = sp.get_figure()
    fig.savefig("./scatter_plot.png")
    dataset_api.upload("./scatter_plot.png", "Resources/images", overwrite=True)

if __name__ == "__main__":
    if LOCAL == True:
        # NOTE: Create an .env file in the root directory if you want to run this locally
        g()
    else:
        modal.runner.deploy_stub(stub)
        
        
        