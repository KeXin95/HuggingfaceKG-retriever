import pandas as pd
import json
import os
from configs import main_config as config

def add_author_col(nodes_df):
    authors_df = pd.read_json(os.path.join(config.JSON_PATH, 
                                           'user_publish_model.json'))
    
    auth_model_dict = dict(zip(authors_df['model_id'], authors_df['user_id']))
    return nodes_df['id'].map(auth_model_dict)
