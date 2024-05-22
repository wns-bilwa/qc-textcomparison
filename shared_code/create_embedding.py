from openai import AzureOpenAI
from rapidfuzz import fuzz
import numpy as np
import os


API_KEY=os.environ.get("api_key")
API_VERSION=os.environ.get("api_version")
AZURE_ENDPOINT=os.environ.get("azure_endpoint")
MODEL=os.environ.get("model")

client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)


def generate_embeddings(text, model=MODEL):  # model = "deployment_name"
    embed = client.embeddings.create(input=text, model=model).data[0].embedding
    return embed


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def fuzzyratio_similarity(a, b):
    print("Qration:", fuzz.QRatio(a, b, score_cutoff=0.1))
    print("Token set ration:", fuzz.partial_token_set_ratio(a, b))
    return fuzz.ratio(a, b)
