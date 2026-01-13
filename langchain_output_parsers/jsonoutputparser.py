print("STARTING SCRIPT")

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


print("IMPORT DONE")

model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    # pipeline_kwargs=dict(temperature=0.7, max_new_tokens=100),
    # return_full_text=False,
)
print("MODEL LOADED, INVOKING...")

parser= JsonOutputParser()
template=PromptTemplate(
    template='Give me the name , age and city of a fictional person.\n {format_instructions}',
    input_variables=[],
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

# prompt=template.format()
# print(prompt)
# # result = model.invoke(prompt)


chain= template|model|parser

# result=model.invoke(prompt)
result=chain.invoke({})
parsed_output=parser.parse(result)
print(parsed_output)

