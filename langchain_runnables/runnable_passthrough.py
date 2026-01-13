from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough

prompt1=PromptTemplate(
    template="Generate a joke on {topic}",
    input_variables=["topic"],
)

prompt2=PromptTemplate(
    template="Explain the following joke {topic}",
    input_variables=["topic"],
)


model = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')\

parser=StrOutputParser();
joke_gen_chain=RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    joke=RunnablePassthrough(),
    explanation=RunnableSequence(prompt2, model, parser)
)

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain
)

final_chain_result = final_chain.invoke({"topic": "artificial intelligence"})
print(final_chain_result)


