from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableSequence, RunnableParallel,RunnablePassthrough, RunnableLambda, RunnableBranch


prompt=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
)

prompt2=PromptTemplate(
    template="Summarize the following text: {text}",
    input_variables=["text"],
)

model=ChatGoogleGenerativeAI(model='models/gemini-2.5-flash')
parser=StrOutputParser();

report_gen_chain=RunnableSequence(prompt, model, parser)

branch_chain=RunnableBranch(
   (lambda x: len(x.split())>500,RunnableSequence(prompt2, model, parser)),
   RunnablePassthrough()
)

final_chain=RunnableSequence(
    report_gen_chain,
    branch_chain
)

final_chain_result = final_chain.invoke({"topic": "The impact of artificial intelligence on modern healthcare systems"})
print(final_chain_result)

