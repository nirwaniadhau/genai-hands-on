from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableSequence, RunnableParallel

prompt1=PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"],
)

prompt2=PromptTemplate(
    template="Generate a linkedin post about {topic}",
    input_variables=["topic"],
)

model = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')\

parser=StrOutputParser();

parallel_chain = RunnableParallel(
    tweet=RunnableSequence(prompt1, model, parser),
    linkedin_post=RunnableSequence(prompt2, model, parser)
)

parallel_chain_result = parallel_chain.invoke({"topic": "Generative ai learning roadmap"})

print(parallel_chain_result)