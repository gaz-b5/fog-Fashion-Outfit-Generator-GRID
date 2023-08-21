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
from flask import Flask, render_template, request
import re
from googlesearch import search
from langchain.document_loaders.csv_loader import CSVLoader
# import aspose.words as aw

chat_history = []
products = ""
# context = "some value"
# sum_bj = "Objective is writing notes for your notebook, you will keep a short but detailed notebook to keep all the information about trends and the data of the user like favourite color, brand, style, purchase history, age, gender, name, ethnicity and all that you have understood about latest fashion trends from the given context. "
load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")


# 1. Tool for search
def search_trends(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query + " site:pinterest.com OR site:instagram.com"
    })

    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    return response.text


def search_products(flipkartQuery):
    products = ""
    url = "https://www.flipkart.com/search?q=" + flipkartQuery + \
        "&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"

    print(url)

    r = requests.get(url)
    print(r)

    # Decode the response content using 'utf-8'
    soup = BeautifulSoup(r.content, "html.parser")
    divs = soup.select('div._1xHGtK._373qXS')

    for x in divs[:6]:
        # Print the text content of each div
        url = "https://www.flipkart.com" + \
            x.find('a', class_='_2UzuFa').get("href")
        brand = x.find('div', class_='_2WkVRV')
        title = x.find('a', class_='IRpwTa')
        img = x.find('img', class_='_2r_T1I')
        price = x.find('div', class_='_30jeq3')
        if bool(img):
            img = img.get("src")
        elif bool(x.find('img', class_="_396cs4 _2amPTt _3qGmMb")):
            img = x.find('img', class_="_396cs4 _2amPTt _3qGmMb").get("src")
        else:
            img = ""
        if bool(brand):
            brand = brand.text
        else:
            brand = ""
        if bool(title):
            title = title.get("title")
        else:
            title = ""
        if bool(price):
            price = price.text.encode(
                'ascii', errors='ignore')
        else:
            price = ""

        product = "Brand: "+brand+" <br> "+title + \
            "<br> Price: "+str(price)+" <br> "+url+" <br> "
        print(brand)
        print(title)
        print(url)
        print(img)
        print(price)
        print()
        if not bool(x.find('div', class_='_2I5qvP')):
            products += product
    return products


# 2. Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
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
        print("CONTENTTTTTT:", text)
        # global context
        # global sum_bj
        if len(text) > 10000:
            output = summary(objective, text)
            # context = context+output
            # context = summary(sum_bj, context)
            return output
        else:
            # context = context+text
            # context = summary(sum_bj, context)
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
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


# get user data from csv
def get_user_data(rubbish):
    if selected_profile == "radio-card-1":
        loader = CSVLoader(file_path="mahuya_data.csv")
        user_data = loader.load()
        print(user_data)
    elif selected_profile == "radio-card-2":
        loader = CSVLoader(file_path="mihir_data.csv")
        user_data = loader.load()
        print(user_data)
    elif selected_profile == "radio-card-3":
        loader = CSVLoader(file_path="myra_data.csv")
        user_data = loader.load()
        print(user_data)
    else:
        user_data = ""
        print(user_data)
    return user_data


# 3. Create langchain agent with the tools above
tools = [
    Tool(
        name="Search_trends",
        func=search_trends,
        description="useful for when you have to search for latest fashion trends"
    ),
    Tool(
        name="Search_products",
        func=search_products,
        description="useful for when you have to product links for products you want to recommend"
    ),
    Tool(
        name="get_user_data",
        func=get_user_data,
        description="useful for when you have to get the user data to understand the preference of the user like preferred style, color choices, and favorite brands better"
    ),
    ScrapeWebsiteTool(),
]


system_message = SystemMessage(
    content="""
Welcome to the Fashion Outfit Generator! I'm here to help you discover the perfect outfit that matches your style. WHEN YOU GET A PROMPT, YOU ABSOLUTELY WILL FOLLOW THESE STEPS IN THIS ORDER:

1. **Getting and Analysing user data**
    Get the user data from the get user data function , (the password is abcd, so put that in it) and this data contains the user's age, gender, name, ethnicity and the products the user already owns. Using this data find what their favourite color is, what their favourite brand is and how the user likes to dress.

2. *Searching for latest fashion trends**
    Use the user's favourite color, brand, style and the prompt given by the user to find generate a search query to search for latest fashion trends relevant to the need of the user. You should include the users favourite color, brand, gender, age in the query to make it more personalised.

3. **Scraping and Analysing results**
    You need to scrape the first 3 post results and analyse them to get a general idea about the latest fashion trends keeping in consideration the necessity of the user. For example if the user already has beige trousers and is asking for a shirt, you should analyse what colored shirt best goes with the beige pants in the fashion trends.

4. **Generating Trendy Recommendations:**
    FOLLOWING THIS STEP AS CLOSELY AS YOU CAN IS VERY IMPORTANT.
    Once you have analsed the user data and fashion trends, prepare a short 6 to 10 word description  of what will be a very good recommendation. This desciption should contain the what you have analysed from the data.
    For exammple: lets supppose you analysed the latest fashion trends relevant to the user and found an x brand y color z fit w type shirt is a very good recommendation, then you this is your description

5. **Finding available products**
    Now search for this personalised fashion recommendation in the search products function with the query as the 5 word description along with the budget specified by the user.

6. **Presenting fashion recommendation**
    Display a maximum of 3 products most relevant to the what you have understood till now.

Please note that you won't be asking any further questions. You are here to work with the information the user has given you and provide the user with the best recommendations based on your preferences and the latest trends."""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=5000)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)


def format_links(text):
    # Find URLs in the text using regular expressions
    urls = re.findall(r'(https?://\S+)', text)
    links = ""
    # Replace each URL with an HTML anchor tag
    for url in urls:
        text = text.replace(url, f'<a href="{url[:-1]}"> [click here] </a>)')
        links = links + "<br>"+url
    return make_cards(text)


def make_cards(content):
    # Find URLs in the text using regular expressions
    urls = re.findall(r"https://www.flipkart\.com[^\s]*", content)
    allcards = """<div class="contain-all-cards">"""
    # Replace each URL with a card
    for url in urls:
        pr = requests.get(url)
        x = BeautifulSoup(pr.content, "html.parser")
        brand = x.find('span', class_='G6XhRU')
        title = x.find('span', class_='B_NuCI')
        img = x.find('img', class_='_2r_T1I _396QI4')
        price = x.find('div', class_='_30jeq3 _16Jk6d')
        if bool(img):
            img = img.get("src")
        else:
            img = ""
        if bool(brand):
            brand = brand.text
        else:
            brand = ""
        if bool(title):
            title = title.text
        else:
            title = ""
        if bool(price):
            price = price.text.encode(
                'ascii', errors='ignore')
        else:
            price = ""

        str_price = str(price)
        text = """	<div class="product-card"><div class="product-tumb"><div class="badge"><a target="_blank" href="""+url+"""<button class="button-27" role="button">BUY</button></a></div><img src="""+img+""" alt=""></div><div class="product-details"><h4><a href="">""" + \
            brand+"""</a></h4><p>"""+title+"""</p><div class="product-bottom-details"><div class="product-price">&#x20B9 """ + \
            str_price[2:-1] + \
            """</div><div class="product-links"></div></div></div></div>"""
        allcards += text
    print(allcards)
    return content+allcards+"</div>"


# 5. Set this as an API endpoint via FastAPI
app = Flask(__name__)


@app.route("/")
def index():
    return render_template('options.html')


@app.route("/chat", methods=["GET", "POST"])
def startChat():
    global selected_profile
    selected_profile = request.args.get('value')
    print(selected_profile)
    # global context
    # context = "Chat started"
    # get_Chat_response("From now on I will give ask you for certain fashion recommendations, you have to get my data from the user data function, this data containes my age, gender, name, ethnicity and the products I already own.Using this data find what my favourite color is, what my favourite brand is and how I like to dress. Tell me my prefered style, color and brand like you have analysed from the data.")
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg + ". Please check for latest fashion trends for good recommendation, and find something which compliments what I already own."
    return get_Chat_response(input)


def get_Chat_response(text):

    # Let's chat for 15 lines
    for step in range(30):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        # query = text
        query = text

        # global context
        # global sum_bj
        # if context:
        #     query = "Notebook: " + context+"   " + text
        # append the new user input tokens to the chat history
        chat_history.append(query)

        # generated a response while limiting the total chat history to 5000 tokens,
        if query:
            result = agent({"input": query})

        fogResponse = result['output']

        print(fogResponse)
        # context = context+fogResponse
        # context = summary(context, sum_bj)
        formatted = format_links(fogResponse)
        simple_string = formatted.replace('\n', "<br>")
        # simple_string = simple_string.replace('\n', "<br>")
        return simple_string


if __name__ == '__main__':
    app.run()
