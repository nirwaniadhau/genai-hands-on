print("STARTING SCRIPT")

from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate


# ----- Load model -----
model = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=150,
        return_full_text=False,
    ),
)

print("MODEL LOADED")


# ----- Define schema -----
class MarsFacts(BaseModel):
    fact_1: str = Field(description="first fact about the topic")
    fact_2: str = Field(description="second fact about the topic")
    fact_3: str = Field(description="third fact about the topic")


parser = PydanticOutputParser(pydantic_object=MarsFacts)


# ----- Prompt -----
template = PromptTemplate(
    template=(
        "Give 3 facts about {topic}.\n"
        "{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

raw = (template | model)
print(raw.invoke({"topic": "Mars"}))

# # ----- Chain -----
# chain = template | model | parser


# # ----- Run -----
# result = chain.invoke({"topic": "Mars"})
# print(result)
# print(result.model_dump())
