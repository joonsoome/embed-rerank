import os
from openai import OpenAI

base_url = os.environ.get("BASE_URL", "http://localhost:9000/v1")
client = OpenAI(base_url=base_url, api_key="dummy")

res = client.embeddings.create(model="text-embedding-ada-002", input=["hello world"]) 
print("embedding length:", len(res.data[0].embedding))
