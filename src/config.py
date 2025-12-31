import os
from dotenv import load_dotenv

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'gold_analysis'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASS'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', 5432)
}