from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt1=PromptTemplate(
    template="Give me detailed explanation about {topic}",
    input_variables=["topic"],
)

prompt2=PromptTemplate(
    template="Summarize the following text in one sentence: {text}",
    input_variables=["text"],
)
model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
parser=StrOutputParser()

chain=prompt1|model|parser|prompt2|model|parser
result=chain.invoke({"topic":"Employment in india"})
print(result)
chain.get_graph().print_ascii()