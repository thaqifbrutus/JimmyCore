from dotenv import load_dotenv
import os

load_dotenv()

# Configuration variables
DATABASE_URL= os.getenv("DATABASE_URL")
AI_API_KEY = os.getenv("GEMINI_AI_API_KEY")