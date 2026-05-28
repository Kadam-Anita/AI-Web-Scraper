from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Template
template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. Extract only the information that directly matches this description: {parse_description}. "
    "2. Do not include extra text, comments, or explanations. "
    "3. If nothing matches, return an empty string ''. "
    "4. Return only the requested data."
)

#  model Ollama
model = ChatOllama(
    model="gemma:2b",
    temperature=0
) 

def parse_with_llama(dom_chunks, parse_description):

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | model

    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):

        response = chain.invoke({
            "dom_content": chunk,
            "parse_description": parse_description
        })

        print(f"Parsed batch {i} of {len(dom_chunks)}")

        parsed_results.append(response.content)

    return "\n".join(parsed_results)