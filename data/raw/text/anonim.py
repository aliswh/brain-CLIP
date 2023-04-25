import pandas as pd
import numpy as np
from hashlib import sha1

f = "~/Desktop/IN-BodyScanData-03_Parsed_Impressions.xlsx"
df = pd.read_excel(f)

encode = lambda s: sha1(s.encode()).hexdigest()
def encode_df(df):
    df["id"] = [encode(v+w) for v,w in zip(df["Dataset"],df["StudyInstanceUID"])]
    df = df.drop(["Dataset","StudyInstanceUID"], axis=1)
    return df
def sample_and_save(df,n=50):
    df.index = df.id
    df = df.drop("id",axis=1)
    df = df.sample(n=n)
    df.to_excel("~/Desktop/nlp_sample.xlsx", encoding='utf8')

sample_and_save(encode_df(df))