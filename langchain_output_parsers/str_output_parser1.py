print("STARTING SCRIPT")

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


print("IMPORT DONE")

model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.7, max_new_tokens=100),
    return_full_text=False,
)
print("MODEL LOADED, INVOKING...")


## first prompt --> detailed report 
template1= PromptTemplate(
    template='write a detailed report on topic:{topic}',
    input_variables=['topic']
)

##second prompt --> summary 

template2= PromptTemplate(
    template='Write a 5 line summary on the following text. \n {text}',
    input_variables=['text']
)


parser= StrOutputParser()

chain= template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'black hole'})
print(result)