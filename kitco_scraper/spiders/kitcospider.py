import scrapy
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from datetime import datetime
from ..items import KitcoItem


class KitcospiderSpider(scrapy.Spider):
    name = "kitcospider"
    allowed_domains = ["kitco.com"]
    
    MAX_PAGES = 200
    ARTICLE_SELECTOR = 'div.DigestNews_newItem__K4a83'

    def __init__(self, max_pages=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pages = int(max_pages) if max_pages else self.MAX_PAGES
        self.total_items_scraped = 0
        self.seen_urls = set()

    def start_requests(self):
        # Use regular Request - our middleware will handle Selenium
        yield scrapy.Request(
            url="https://www.kitco.com/news/digest#metals",
            callback=self.parse,
            meta={'wait_time': 5}
        )

    def parse(self, response):
        driver = response.meta['driver']
        actions = ActionChains(driver)

        page_count = 1

        while page_count <= self.max_pages:
            self.logger.info(f"Scraping page {page_count}/{self.max_pages}...")

            articles = driver.find_elements(By.CSS_SELECTOR, self.ARTICLE_SELECTOR)
            self.logger.info(f"Found {len(articles)} articles on page {page_count}")

            new_articles = 0
            for article in articles:
                item = self.extract_article(article)
                if item:
                    yield item
                    new_articles += 1
                    self.total_items_scraped += 1

            self.logger.info(f"Scraped {new_articles} new articles from page {page_count}")

            if not self.click_load_more(driver, actions):
                self.logger.info("No more pages to load. Stopping.")
                break
                
            page_count += 1

    def extract_article(self, article):
        try:
            link_element = article.find_element(By.CSS_SELECTOR, 'a.grow')
            url = link_element.get_attribute('href')

            if url in self.seen_urls:
                return None
            self.seen_urls.add(url)

            item = KitcoItem()
            item['url'] = url
            item['headline'] = self.get_headline(article)
            item['timestamp'], item['datetime'] = self.get_timestamp(article)
            
            return item

        except NoSuchElementException as e:
            self.logger.warning(f"Missing element in article: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error extracting article: {e}")
            return None

    def get_headline(self, article):
        try:
            h5 = article.find_element(By.CSS_SELECTOR, 'h5.line-clamp-1')
            headline = h5.get_attribute('textContent')
            if headline:
                return headline.strip()
        except NoSuchElementException:
            pass
        # Fallback: try getting text from the link itself
        try:
            h5 = article.find_element(By.TAG_NAME, 'h5')
            headline = h5.get_attribute('textContent')
            if headline:
                return headline.strip()
        except NoSuchElementException:
            pass

        text = article.get_attribute('textContent')
        if text:
            return text.strip().split('\n')[0]
        return "No headline"    

    def get_timestamp(self, article):
        try:
            p = article.find_element(By.CSS_SELECTOR, 'p.text-gray-500')
            text = p.get_attribute('textContent')
            if text:
                text = text.strip()
                try:
                    parsed = datetime.strptime(text, "%b %d, %Y %I:%M%p")
                    return text, parsed.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    try:
                        parsed = datetime.strptime(text, "%b %d, %Y")
                        return text, parsed.strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        return text, None
        except NoSuchElementException:
            pass
        return None, None

    def click_load_more(self, driver, actions):
        # Click the Load MOre News button
        try:
            import time
            current_count = len(driver.find_elements(By.CSS_SELECTOR, self.ARTICLE_SELECTOR))
            
            load_more_btn = driver.find_element(
                By.CSS_SELECTOR, 
                '.float-right.rounded-md.py-3'
            )
            # Scroll to button
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
                load_more_btn
            )
            time.sleep(1)

            #Click
            driver.execute_script("arguments[0].click();", load_more_btn)
            self.logger.info("Clicked Load More button (JS click)")

            # Wait for new content to load (up to 10 seconds)
            for _ in range(10):
                time.sleep(1)
                new_count = len(driver.find_elements(By.CSS_SELECTOR, self.ARTICLE_SELECTOR))
                if new_count > current_count:
                    self.logger.info(f"New articles loaded: {current_count} -> {new_count}")
                    return True
        
            self.logger.info(f"No new articles loaded after click (still {current_count})")
            return False
            
        except NoSuchElementException:
            self.logger.info("load MOre button not found")
            return False
        except Exception as e:
            self.logger.warning(f"Error clicking Load More: {e}")
            return False

    def closed(self, reason):
        self.logger.info("=" * 50)
        self.logger.info(f"SPIDER CLOSED: {reason}")
        self.logger.info(f"Total items scraped: {self.total_items_scraped}")
        self.logger.info(f"Unique URLs seen: {len(self.seen_urls)}")
        self.logger.info("=" * 50)

                     
            
                
               
