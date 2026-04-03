from src.news.encoder import FinBERTEncoder

encoder = FinBERTEncoder()
encoder.encode_and_save(
    cleaned_parquet=r"C:\Users\24106\PycharmProjects\CSCI-566_Term-Proj_QianTick\data\processed\news\news_cleaned.parquet",
    output_parquet=r"C:\Users\24106\PycharmProjects\CSCI-566_Term-Proj_QianTick\data\features\news_article_embeddings.parquet",
    batch_size=64,
    checkpoint_every=10000,
)