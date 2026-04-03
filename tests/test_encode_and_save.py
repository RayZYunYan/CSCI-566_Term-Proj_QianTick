from src.news import FinBERTEncoder

encoder = FinBERTEncoder()
x = encoder.encode(["Apple beats Q3 earnings"])
print(x.shape)