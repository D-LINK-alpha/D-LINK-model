import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from transformers import ElectraModel, ElectraTokenizer

class Prediction():
    def __init__(self):
        super().__init__()
        self.model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")  # KoELECTRA-Base-v3
        self.tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    def get_cosine_similarity(self, prompt, document):
        return float(cosine_similarity(prompt, document)[0][0])
    
    def get_corrcoef_similarity(self, prompt, document):
        return float(np.corrcoef(prompt, document)[0][1])
    
    def get_similarities(self, prompt, documents, sim_mode):
        similarities = []
        prompt_embedding = self.get_embedding(prompt)
        for d in documents:
            d_embedding = self.get_embedding(d.document)
            if sim_mode == "cos":
                similarities.append({
                    "id": d.id,
                    "similarity": self.get_cosine_similarity(prompt_embedding, d_embedding)
                    })
            elif sim_mode == "cor":
                similarities.append({
                    "id": d.id,
                    "similarity": self.get_corrcoef_similarity(prompt_embedding, d_embedding)
                    })
        return similarities