from scrapy import signals
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from scrapy.http import HtmlResponse
import logging

logger = logging.getLogger(__name__)


class SeleniumMiddleware:
    """Custom Selenium middleware compatible with Selenium 4.x"""
    
    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls(crawler.settings)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        return middleware
    
    def __init__(self, settings):
        chrome_path = settings.get('SELENIUM_DRIVER_EXECUTABLE_PATH')
        arguments = settings.get('SELENIUM_DRIVER_ARGUMENTS', [])
        
        logger.info(f"Initializing Selenium with ChromeDriver: {chrome_path}")
        
        options = Options()
        for arg in arguments:
            options.add_argument(arg)
        
        service = Service(executable_path=chrome_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        logger.info("Selenium WebDriver initialized successfully")
    
    def process_request(self, request, spider):
        logger.debug(f"Selenium fetching: {request.url}")
        self.driver.get(request.url)
        
        # Wait if specified
        wait_time = request.meta.get('wait_time', 5)
        if wait_time:
            import time
            time.sleep(wait_time)
        
        body = self.driver.page_source
        
        # Create response without meta argument
        response = HtmlResponse(
            url=self.driver.current_url,
            body=body.encode('utf-8'),
            encoding='utf-8',
            request=request
        )
        
        # Attach driver to request.meta so spider can access it
        request.meta['driver'] = self.driver
        
        return response
    
    def spider_closed(self, spider):
        logger.info("Closing Selenium WebDriver")
        if self.driver:
            self.driver.quit()