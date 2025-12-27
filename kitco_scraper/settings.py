from shutil import which
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

#Scrapy settings
BOT_NAME = "kitco_scraper"
SPIDER_MODULES = ["kitco_scraper.spiders"]
NEWSPIDER_MODULE = "kitco_scraper.spiders"

ADDONS = {}

ROBOTSTXT_OBEY = False
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
FEED_EXPORT_ENCODING = "utf-8"

# RATE LIMITING & THROTTLING
CONCURRENT_REQUESTS_PER_DOMAIN = 1
CONCURRENT_REQUESTS = 1
DOWNLOAD_DELAY = 2
DOWNLOAD_TIMEOUT = 180

AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 5

RETRY_ENABLED = True
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]

SELENIUM_DRIVER_NAME = "chrome"
SELENIUM_DRIVER_EXECUTABLE_PATH = r'C:\chromedriver\chromedriver.exe'

HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"

SELENIUM_DRIVER_ARGUMENTS = [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-blink-features=AutomationControlled',
    '--ignore-certificate-errors',  # Helps with SSL issues
    '--ignore-ssl-errors',  # Helps with SSL issues
    '--allow-insecure-localhost',  # Helps with SSL issues
    '--disable-web-security',  # Use with caution
]

if HEADLESS:
    SELENIUM_DRIVER_ARGUMENTS.insert(0, '--headless')

ITEM_PIPELINES = {
    "kitco_scraper.pipelines.DuplicatesPipeline": 100,
    'kitco_scraper.pipelines.ValidationPipeline': 200,      
    'kitco_scraper.pipelines.TimestampPipeline': 300,       
    'kitco_scraper.pipelines.NewsCategoryPipeline': 400,    
    'kitco_scraper.pipelines.SentimentPipeline': 500, 
    'kitco_scraper.pipelines.PriceTargetExtractorPipeline': 550,
    'kitco_scraper.pipelines.MarketUrgencyPipeline': 560,
    'kitco_scraper.pipelines.ExpertSourceExtractorPipeline': 570,
    'kitco_scraper.pipelines.PostgreSQLPipeline': 600, 
}
DOWNLOADER_MIDDLEWARES = {
    'kitco_scraper.middlewares.SeleniumMiddleware': 800,
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': 400,
}


