import os
import logging
from datetime import datetime


LOG_FILE = datetime.now().strftime("%Y-%m-%d") + ".log"
logs_dir = os.path.join(os.getcwd(), "logs",datetime.now().strftime("%Y-%m-%d"))

os.makedirs(logs_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s'
)



