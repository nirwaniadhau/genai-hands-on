from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')

result =model.invoke("What is the capital of India?")
print(result.content)


# from google.genai import Client

# client = Client()
# for m in client.models.list():
#     print(m.name)