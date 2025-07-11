from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

llm = ChatOpenAI(openai_api_key="API-Key", model_name="gpt 4o-mini", temperature=0.7, max_tokens=1000)

symptom_parser_template = """
Given the following free-text description of a patient's symptoms, extract the structured list of medical symptoms.

Text: "{text}"
Extracted Symptoms (in list form):
"""

parser_prompt = PromptTemplate(
    input_variables=["text"],
    template=symptom_parser_template
)

symptom_parser_chain = LLMChain(llm=llm, prompt=parser_prompt)

def extract_symptoms_from_text(text):
    response = symptom_parser_chain.run(text)
    try:
        return eval(response.strip())  
    except:
        return []
