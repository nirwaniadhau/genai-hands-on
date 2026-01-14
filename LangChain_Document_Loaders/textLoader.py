from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
parser=StrOutputParser()
prompt=PromptTemplate(
    template="Write a summary of the following poem -\n {poem}",
    input_variables=['poem']
)

loader = TextLoader(r"D:\Gen_Ai\LangChain_Document_Loaders\cricket.txt",encoding='utf-8')
docs = loader.load()

chain=prompt|model|parser
result=chain.invoke({"poem": docs[0].page_content})
print(result)

print(type(docs))

print(docs[0])
print(type(docs[0]))
print(docs[0].page_content)
print(docs[0].metadata)