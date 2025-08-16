import os
import json
from google import genai
from google.genai.types import GenerateContentConfig
import re

GEMINI_MODEL_DEFAULT = "gemini-2.5-pro"
# api_key = os.getenv("GEMINI_API_KEY")
api_key = "AIzaSyD8d74JXRIoAMctsgaJL3SfxSaDXKcf5ys"

"AIzaSyC-aZloVBPJby_B5LcgAo_XIWqolS3PELw"

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model=GEMINI_MODEL_DEFAULT,
    contents="Write a short python code for getting 20 random numbers and their mean",
    config=GenerateContentConfig(temperature=0)
)
print(response.candidates[0].content.parts[0].text.strip())

