from googlesearch import search

# Define the search query
query = "traditional outfit latest fdashion trends"

# Set the number of search results to retrieve
num_results = 10

# Perform the search and retrieve the results
search_results = search(query, num_results=num_results, lang='en')

# Iterate over the search results
for i, result in enumerate(search_results, start=1):
    # print(f"Result {i}:")
    print(result)

    # # Retrieve the page title
    # try:
    #     import requests
    #     from bs4 import BeautifulSoup

    #     # Send a GET request to the URL
    #     response = requests.get(result)

    #     # Parse the HTML content
    #     soup = BeautifulSoup(response.text, 'html.parser')

    #     # Retrieve the page title
    #     page_title = soup.title.string

    #     print("Page Title:", page_title)
    # except:
    #     print("Failed to retrieve page title.")

    print()
