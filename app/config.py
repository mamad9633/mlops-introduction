import os

from dotenv import load_dotenv

load_dotenv()

LOGISTIC_MODEL = os.getenv("LOGISTIC_MODEL")
RF_MODEL = os.getenv("RF_MODEL")

API_KEY = os.getenv("API_KEY")