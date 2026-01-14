from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
parser=StrOutputParser()
prompt=PromptTemplate(
    template='Answer the following \n {question} from the following text \n {text}',
    input_variables=['question','text']
)
url='https://indianexpress.com/'
loader=WebBaseLoader(url)


docs=loader.load()
# print(len(docs))
# print(docs[0].page_content)

chain=prompt|model|parser
result=chain.invoke({'question':'What is this new about', 'text':docs[0].page_content})
print(result)