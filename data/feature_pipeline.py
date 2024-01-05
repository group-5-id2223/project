import feature_library as fl

RUN_ON_MODAL = False
MAX_CRAWL_TIME = 2*60*60

def get_live_data():
    #TODO
    pass

def g():
    import os
    import numpy as np
    import pandas as pd
    import hopsworks
    from glob import glob
    from tqdm import tqdm
    
    from feature_processing import load_text_encoder
    
    tsvs = glob('./csvs/*.tsv')
    dfs = pd.concat([pd.read_csv(tsv, sep='\t', usecols=lambda x: x!='id') for tsv in tsvs])
    dfs['url'].fillna('', inplace=True)
    
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    feat_grp = fs.get_or_create_feature_group(
        name="hackernews-posts",
        version=1,
        primary_key=['time'],
        description="Hackernews posts from API",
    )
    
    feat_grp.insert(dfs, write_options={"wait_for_job": True})


import modal
stub = modal.stub()
image = modal.Image.debian_slim().pip_install(['hopsworks', 'transformers', 'torch', 'pandas', 'einops'])
@stub.function(image=image,
               schedule=modal.Period(days=1),
               secret=modal.Secret.from_name("hackernews-score-pred")
               )
def f():
    g()


if __name__ == '__main__':
    
    if RUN_ON_MODAL:
        with stub.run():
            f.call()
    else:
        g()
    # stories = fl.get_last_ten_stories()
    # data: list = [fl.process_story(story) for story in stories]
    # print(data)