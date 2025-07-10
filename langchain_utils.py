import os
import ast
from typing import List
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# Use environment variable for API key
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    model_name=os.getenv("OPENAI_MODEL_NAME", "gpt 4o-mini"),
    temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
)

symptom_parser_template = """
Given the following free-text description of a patient's symptoms, extract the structured list of medical symptoms.

Text: "{text}"
Extracted Symptoms (in list form):
Always return a valid Python list as your output, and nothing else. Do not include explanations or extra text.
"""

parser_prompt = PromptTemplate(
    input_variables=["text"],
    template=symptom_parser_template
)

symptom_parser_chain = LLMChain(llm=llm, prompt=parser_prompt)

def extract_symptoms_from_text(text: str) -> List[str]:
    """
    Extracts a list of symptoms from free-text using an LLM chain.
    Args:
        text (str): Free-text description of symptoms.
    Returns:
        List[str]: List of extracted symptoms.
    """
    response = symptom_parser_chain.run(text)
    try:
        return ast.literal_eval(response.strip())
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing response: {e}\nResponse: {response}")
        return []
