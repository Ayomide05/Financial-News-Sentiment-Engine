# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field

class KitcoItem(scrapy.Item):
    # Basic Fields
    headline = Field()
    timestamp = Field()
    datetime = Field()
    url = Field()

    # For Pipeline processing
    date = Field()
    time = Field()
    categories = Field()
    primary_category = Field()
    category_confidence = Field()
    sentiment = Field()
    sentiment_score = Field()
    sentiment_conflict = Field()

    # Urgency fields (added for pipelines)
    urgency_score = Field()
    urgency_category = Field()
    is_breaking_news = Field()
    time_horizon = Field()
    market_flags = Field()
    
    # Price analysis fields
    price_targets = Field()
    avg_price_target = Field()
    
    # Expert analysis fields
    expert_mentions = Field()
    source_credibility = Field()

class KitcoArticleItem(scrapy.Item):
    #From original scrape
    headline = Field()
    url = Field()
    list_timestamp = Field()
    
    #New fields from full article
    full_text = Field()
    published_date = Field()
    updated_date = Field()
    author = Field()
    tags = Field()

    published_datetime = Field()
    published_date_only = Field()
    published_time_only = Field()
    updated_datetime = Field()
    updated_date_only = Field()
    updated_time_only = Field()
    categories = Field()
    primary_category = Field()
    category_confidence = Field()
    sentiment = Field()
    sentiment_score = Field()
    headline_sentiment = Field()
    headline_sentiment_score = Field()
    content_sentiment = Field()
    content_sentiment_score = Field()
    sentiment_conflict = Field()
    sentiment_details = Field()

    #Price analysis
    price_mentions = Field()
    price_targets = Field()
    support_levels = Field()
    resistance_levels = Field()
    avg_price_target = Field()

    #Urgency Analysis
    urgency_score = Field()
    urgency_category = Field()
    is_breaking_news = Field()
    time_horizon = Field()
    market_flags = Field()

    # Expert analysis
    expert_sources = Field()
    source_credibility = Field()
    expert_quotes = Field()

    
