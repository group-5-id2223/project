import feature_library as fl

RUN_ON_MODAL = True
MAX_CRAWL_TIME = 2 * 60 * 60

def g():
    import os
    import numpy as np
    import pandas as pd
    import hopsworks
    from tqdm import tqdm
    
    from feature_processing import load_text_encoder
    


import modal
stub = modal.stub()
image = modal.Image.debian_slim().pip_install(['hopsworks'])

if __name__ == '__main__':
    stories = fl.get_last_ten_stories()
    data: list = [fl.process_story(story) for story in stories]
    print(data)