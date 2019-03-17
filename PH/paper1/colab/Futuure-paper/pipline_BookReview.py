#! /usr/bin/python3.6
import text_processing as tp
import Utils as u

import pandas as pd
import json

with open('/media/fsg/74C86089C8604C04/download/reviews_Books_5.json', 'r') as f:
    reviews = f.readlines()
    data = [json.loads(item.strip('\n')) for item in reviews]
    df = pd.DataFrame(data)
    df.to_csv('Amazon_Data_Frame.csv')


