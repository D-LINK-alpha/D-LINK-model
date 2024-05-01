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
        # 텍스트를 토큰화하고 토큰의 ID로 변환
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # 모델에 입력하여 hidden state 출력 (문맥적 임베딩)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 마지막 hidden state의 첫번째 토큰 (CLS 토큰)을 문서의 임베딩으로 사용
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
    
    def get_cosine_similarity(self, embeddings):
        similarities = []
        prompt = embeddings[0]
        for i in range(1, len(embeddings)):
            similarities.append(cosine_similarity(prompt, embeddings[i])[0][0])
        return similarities
    
    def get_corrcoef_similarity(self, embeddings):
        similarities = []
        prompt = embeddings[0]
        for i in range(1, len(embeddings)):
            similarities.append(np.corrcoef(prompt, embeddings[i])[0][1])
        return similarities