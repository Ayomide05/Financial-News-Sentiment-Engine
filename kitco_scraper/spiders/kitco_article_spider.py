import scrapy
from scrapy_selenium import SeleniumRequest
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import psycopg2
import os
from dotenv import load_dotenv
from ..items import KitcoArticleItem

# Load environment variable from .env file
load_dotenv()

class KitcoArticleSpider(scrapy.Spider):
    name = "kitco_articles"
    allowed_domains = ["kitco.com"]

    SELECTORS = {
        'body': "#articleBody p",
        'time': "div.flex.items-center.md\\:block time",
        'author': "div.pl-3.md\\:pl-0 a",
        'tags': "div.flex.flex-wrap.gap-2 span"
    }

    def __init__(self, limit=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = int(limit) if limit else None
        self.articles_processed = 0
        self.articles_failed = 0

    def start_requests(self):
        articles = self.get_pending_articles()
        self.logger.info(f"FOund {len(articles)} articles to scrape")

        for url, headline in articles:
            yield SeleniumRequest(
                url=url,
                callback=self.parse_article,
                meta={
                    'headline': headline,
                    'url': url
                },
                wait_time = 3,
                dont_filter = True
            )
    def get_pending_articles(self):
        """Fetch articles that haven't been fully scraped yet"""
        conn = None
        cur = None

        try:
            conn = psycopg2.connect(
                dbname=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASS"),
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT", 5432)
            )
            cur = conn.cursor()

            # Only get articles without full test
            query = """
                SELECT url, headline FROM articles
                WHERE full_text IS NULL OR full_text = ''
                ORDER BY Published_date DESC
            """

            if self.limit:
                query += f" LIMIT {self.limit}"
            
            cur.execute(query)
            return cur.fetchall()
        except psycopg2.Error as e:
            self.logger.error(f"Database error: {e}")
            return []
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def parse_article(self, response):
        driver = response.meta['driver']

        item = KitcoArticleItem()
        item['headline'] = response.meta.get('headline')
        item['url'] = response.meta.get('url')

        # Extract all fields
        item['full_text'] = self.extract_full_text(driver)
        item['published_date'], item['updated_date'] = self.extract_dates(driver)
        item['author'] = self.extract_author(driver)
        item['tags'] = self.extract_tags(driver)

        # Track sucess/failure
        if item['full_text']:
            self.articles_processed += 1
        else:
            self.articles_failed += 1
            self.logger.warning(f"No content extracted for: {item['url']}")
        
        yield item

    def extract_full_text(self, driver):
        """Extract article body text"""
        try:
            paragraphs = driver.find_elements(By. CSS_SELECTOR, self.SELECTORS['body'])
            text = "\n".join([p.get_attribute('textContent') for p in paragraphs if p.get_attribute('textContent').strip()])
            return text if text else None
        except Exception as e:
            self.logger.error(f"Error extracting full text: {e}")
            return None
        
    def extract_dates(self, driver):
        """Extract pulished and updated dates"""
        try:
            time_tags = driver.find_elements(By.CSS_SELECTOR, self.SELECTORS['time'])
            published = time_tags[0].get_attribute('textContent') if len(time_tags) > 0 else None
            updated = time_tags[1].get_attribute('textContent') if len(time_tags) > 1 else None
            return published, updated
        except Exception as e:
            self.logger.error(f"Error extracting dates: {e}")
            return None, None
        
    def extract_author(self, driver):
        """Extract author name"""
        try:
            return driver.find_element(By.CSS_SELECTOR, self.SELECTORS['author']).get_attribute('textContent')
        except NoSuchElementException:
            return None
        except Exception as e:
            self.logger.error(f"Error, extracting author: {e}")
            return None
        
    def extract_tags(self,driver):
        """Extract article tags"""
        try:
            tag_elements = driver.find_elements(By.CSS_SELECTOR, self.SELECTORS['tags'])
            return [tag.get_attribute('textContent') for tag in tag_elements if tag.get_attribute('textContent').strip()]
        except Exception as e:
            self.logger.error(f"Error extracting tags: {e}")
            return []
        
    def closed(self, reason):
        """Summary Statistics"""
        self.logger.info("=" * 50)
        self.logger.info(f"ARTICLE SPIDER CLOSED: {reason}")
        self.logger.info(f"Successfully processed: {self.articles_processed}")
        self.logger.info(f"Failed to extract: {self.articles_failed}")
        self.logger.info("=" * 50)



