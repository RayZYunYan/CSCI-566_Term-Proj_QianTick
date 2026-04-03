import torch
from src.news.encoder import FinBERTEncoder

# python -m scripts.test_mock_encoder

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

encoder = FinBERTEncoder(device=device)

test_headlines = [
    "Apple beats Q3 earnings",
    "NVIDIA announces new chip",
]

output = encoder.encode(test_headlines)

print(output.shape)
