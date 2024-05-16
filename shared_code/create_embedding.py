from openai import AzureOpenAI
from rapidfuzz import fuzz
import numpy as np


client = AzureOpenAI(
    api_key="c46e23bb620d4b1cb1af0cfa070d31a4",
    api_version="2024-02-01",
    azure_endpoint="https://wts-opensearch-gpt.openai.azure.com/",
)


def generate_embeddings(text, model="embedding"):  # model = "deployment_name"
    embed = client.embeddings.create(input=text, model=model).data[0].embedding
    return embed


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def fuzzyratio_similarity(a, b):
    print("Qration:", fuzz.QRatio(a, b, score_cutoff=0.1))
    print("Token set ration:", fuzz.partial_token_set_ratio(a, b))
    return fuzz.ratio(a, b)
