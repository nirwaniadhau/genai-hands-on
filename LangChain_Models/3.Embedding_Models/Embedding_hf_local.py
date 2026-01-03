from langchain_huggingface import HuggingFaceEmbeddings
print("IMPORT DONE")
embedding =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text="Delhi is the capital of India."
vector=embedding.embed_query(text)
print(f"Text: {text}\nVector: {str(vector)}")
print("EMBEDDING DONE")
