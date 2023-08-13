import os
from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# 1. Tool for search


def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print("search results: ")
    print(response.text)

    return response.text


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website..." + url)
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text.encode('ascii', errors='ignore'))
        return text.encode('ascii', errors='ignore')
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents(
        [content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when need to search for latest fashion trends or have to get product links for the user. Search for product which will be a good fashion recommendation to the user."
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""Welcome to the Fashion Outfit Generator powered by OpenAI's GPT-3.5! I'm here to help you discover the perfect traditional outfit that matches your style. Here's how I work:

1. **Searching for Latest Fashion Trends:**
   I'll use the information you provide to create a tailored search query. This query will be used to search Google, focusing on social media platforms like Instagram, Twitter, and TikTok. I'll look for the latest traditional outfit trends that match your style and preferences. I'll make sure to format the query such that the results are a way of helping m develop the fashion sense and not just direct product. for example: Trendy fashion ideas with x outfit for y years old and for z gender.

2. **Scraping and Analyzing Results:**
   From the search results, I'll scrape the content of only instagram and tiktok websites. This will help me analyze and understand the latest traditional outfit trends that are currently in vogue for boys. I'll ensure that the recommendations I provide are trendy and align with your preferences.

3. **Generating Trendy Recommendations:**
   Based on my analysis, I'll suggest a set of traditional outfit recommendations for you. These recommendations may include clothing, accessories, and more, ensuring that they are in line with both your preferences and the latest trends. I'll aim to provide up to 3 top-notch suggestions.
   I WILL NOT generate recommendations before scraping and analysing latest fashion trend.I WILL NOT generate recommendations before scraping and analysing latest fashion trend.

4. **Finding Available Products:**
   One of the most crucial rules is to always provide product links, no matter what. I'll make sure to search for these traditional outfits on popular e-commerce websites such as Amazon, Flipkart, Nykaa, Myntra, and others. I'll ensure that the links I provide lead directly to product pages, allowing you to explore and purchase them hassle-free.

5. **Presenting Your Recommendations:**
   I'll display your personalized traditional outfit recommendations along with first 3 clickable links that take you directly to the product pages on the respective e-commerce websites.
    In WILL also explain why these are trendy.

Please note that I won't be asking any further questions. I'm here to work with the information you've given me and provide you with the best recommendations based on your preferences and the latest trends. Let's start by creating the perfect traditional outfit for you!
"""
)


agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


# 4. Use streamlit to create a web app
def main():
    st.set_page_config(page_title="FOG", page_icon=":bird:")

    st.header("Fashion Outfit generator :bird:")
    query = st.text_input("User prompt")

    if query:
        st.write("Doing research for ", query)

        result = agent({"input": query})

        st.info(result['output'])


if __name__ == '__main__':
    main()


# 5. Set this as an API endpoint via FastAPI
# app = FastAPI()


# class Query(BaseModel):
#     query: str


# @app.post("/")
# def researchAgent(query: Query):
#     query = query.query
#     content = agent({"input": query})
#     actual_content = content['output']
#     return actual_content
