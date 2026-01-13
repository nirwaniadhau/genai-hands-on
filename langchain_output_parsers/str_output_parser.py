print("STARTING SCRIPT")

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

print("IMPORT DONE")

model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(temperature=0.7, max_new_tokens=100),
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

prompt1 = template1.invoke({'topic':'Climate Change'})
result = model.invoke(prompt1)
# print(result)

prompt2 = template2.invoke({'text':result})
summary_result = model.invoke(prompt2)
print(summary_result)