import logging as log
import dotenv
import sys
import os

dotenv.load_dotenv()

log_level = os.environ.get("LOG_LEVEL", "DEBUG")
formatter = log.Formatter('[%(asctime)s] [%(process)d] [%(thread)d] [%(levelname)s] %(name)s: %(message)s')
stream_handler = log.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logging = log.getLogger(__name__)
logging.addHandler(stream_handler)
logging.setLevel(log_level)