from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

def scrape_news():
    # Set up Chrome options to ignore SSL errors
    chrome_options = Options()
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')

    # Set up Selenium with ChromeDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    # Open Yahoo Finance
    driver.get('https://finance.yahoo.com')

    # Wait for a few seconds to ensure JavaScript loads
    time.sleep(5)

    # Get page source and parse with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Close the Selenium browser
    driver.quit()

    # Extract news headlines
    headlines = soup.find_all('h3', {'class': 'Mb(5px)'})

    # Print each headline text
    for headline in headlines:
        print(headline.text)

# Call the function
scrape_news()
