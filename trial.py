import urllib.request
from bs4.element import Comment
import requests
from bs4 import BeautifulSoup


def scrape_website(url):
    # Send a GET request to the website
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all the text elements in the HTML
        text_elements = soup.find_all(text=True)

        # Filter out unwanted elements (e.g., script, style, etc.)
        legible_text = filter(is_legible, text_elements)

        # Join the legible text elements into a single string
        result = ' '.join(legible_text)

        return result

    # If the request was not successful, return None
    return None


def is_legible(element):
    # Filter out unwanted elements based on their tag name or class
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if element.parent.name == 'a' and element.parent.get('href'):
        return False
    return True


# Example usage
url = 'https://www.vogue.in/fashion/fashion-trends'
legible_text = scrape_website(url)
print(legible_text)


# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True


# def text_from_html(body):
#     soup = BeautifulSoup(body, 'html.parser')
#     texts = soup.findAll(text=True)
#     visible_texts = filter(tag_visible, texts)
#     return u" ".join(t.strip() for t in visible_texts)


# html = urllib.request.urlopen(
#     'https://www.vogue.in/fashion/fashion-trends').read()
# print(text_from_html(html))
