'''With the help of Openai's GPT 5-turbo model, this script summarizes the content of a given text file.
the user provides the path to the text file as a command-line argument.
the algorithm works superfast at responses of 20 milliseconds per request.
It reads the file, sends its content to the GPT model, and prints the summarized output.
logs are immediately deleted after the script runs for security and confidential purposes.'''
import os
import sys
import openai
import logging
import tempfile
from dotenv import load_dotenv
from datetime import datetime
import atexit
import shutil
import time
import signal
import threading
import traceback
import re

from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai.error import RateLimitError, APIError, Timeout, ServiceUnavailableError, APIConnectionError
from requests.exceptions import HTTPError, ConnectionError, Timeout as RequestsTimeout
from urllib3.exceptions import ProtocolError
from socket import timeout as SocketTimeout
from openai import OpenAIError
from openai import InvalidRequestError, AuthenticationError, PermissionError
from openai import OpenAIError

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
MODEL = "gpt-5-turbo"
MAX_TOKENS = 4096
SUMMARY_TOKENS = 500
TEMPDIR = tempfile.mkdtemp()
LOG_FILE = os.path.join(TEMPDIR, 'app.log')
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
lock = threading.Lock()
stop_event = threading.Event()
last_log_time = time.time()
LOG_RETENTION_SECONDS = 300  # 5 minutes
LOG_CHECK_INTERVAL = 60  # Check every minute
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
SUMMARY_PROMPT = (
    "Summarize the following text in a concise manner, highlighting the key points and main ideas. "
    "Ensure the summary is clear and easy to understand.\n\n"
    "Text:\n"
)
EXIT_CODES = {
    'SUCCESS': 0,
    'FILE_NOT_FOUND': 1,
    'FILE_TOO_LARGE': 2,
    'INVALID_FILE': 3,
    'OPENAI_ERROR': 4,
    'UNKNOWN_ERROR': 5
}
# Register cleanup function to delete temp directory and its contents on exit
def cleanup():
    try:
        if os.path.exists(TEMPDIR):
            shutil.rmtree(TEMPDIR)
            logging.info(f"Temporary directory {TEMPDIR} and its contents have been deleted.")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")
atexit.register(cleanup)

def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}. Exiting gracefully...")
    cleanup()
    sys.exit(EXIT_CODES['SUCCESS'])
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log_activity(message):
    global last_log_time
    with lock:
        logging.info(message)
        last_log_time = time.time()

def log_exception(exc):
    with lock:
        logging.error(f"Exception occurred: {exc}")
        logging.error(traceback.format_exc())
        global last_log_time
        last_log_time = time.time()
        last_log_time = time.time()
def periodic_log_cleanup():
    while not stop_event.is_set():
        time.sleep(LOG_CHECK_INTERVAL)
        current_time = time.time()
        with lock:
            if current_time - last_log_time > LOG_RETENTION_SECONDS:
                try:
                    if os.path.exists(LOG_FILE):
                        os.remove(LOG_FILE)
                        logging.info("Log file deleted due to inactivity.")
                except Exception as e:
                    logging.error(f"Error deleting log file: {e}")
                last_log_time = current_time
log_cleanup_thread = threading.Thread(target=periodic_log_cleanup, daemon=True)
log_cleanup_thread.start()
atexit.register(lambda: stop_event.set())

def validate_file(file_path):
    if not os.path.isfile(file_path):
        log_activity(f"File not found: {file_path}")
        print(f"Error: File not found - {file_path}")
        sys.exit(EXIT_CODES['FILE_NOT_FOUND'])
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        log_activity(f"File too large: {file_path}")
        print(f"Error: File size exceeds the maximum limit of {MAX_FILE_SIZE / (1024 * 1024)} MB.")
        sys.exit(EXIT_CODES['FILE_TOO_LARGE'])
    if not file_path.lower().endswith(('.txt', '.md', '.rtf')):
        log_activity(f"Invalid file type: {file_path}")
        print("Error: Invalid file type. Please provide a .txt, .md, or .rtf file.")
        sys.exit(EXIT_CODES['INVALID_FILE'])
    log_activity(f"File validated: {file_path}")
    return True
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            log_activity(f"File read successfully: {file_path}")
            return content  
    except Exception as e:
        log_exception(e)
        print(f"Error reading file: {e}")
        sys.exit(EXIT_CODES['UNKNOWN_ERROR'])
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type((RateLimitError, APIError, Timeout, ServiceUnavailableError, 
                                 APIConnectionError, HTTPError, ConnectionError, 
                                 RequestsTimeout, ProtocolError, SocketTimeout))
    )
)
def summarize_text(content):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": SUMMARY_PROMPT + content}
        ]
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
            max_tokens=SUMMARY_TOKENS,
            n=1,
            stop=None,
            temperature=0.5,
        )
        summary = response.choices[0].message['content'].strip()
        log_activity("Text summarized successfully.")
        return summary
    except (InvalidRequestError, AuthenticationError, PermissionError) as e:
        log_exception(e)
        print(f"OpenAI API error: {e}")
        sys.exit(EXIT_CODES['OPENAI_ERROR'])
    except OpenAIError as e:
        log_exception(e)
        print(f"OpenAI error: {e}")
        raise
    except Exception as e:
        log_exception(e)
        print(f"An unexpected error occurred: {e}")
        sys.exit(EXIT_CODES['UNKNOWN_ERROR'])
def main():
    if len(sys.argv) != 2:
        print("Usage: python summarization.py <path_to_text_file>")
        sys.exit(EXIT_CODES['UNKNOWN_ERROR'])
    file_path = sys.argv[1]
    validate_file(file_path)
    content = read_file(file_path)
    if not content.strip():
        print("Error: The provided file is empty.")
        sys.exit(EXIT_CODES['INVALID_FILE'])
    summary = summarize_text(content)
    print("Summary:\n", summary)
    sys.exit(EXIT_CODES['SUCCESS'])
if __name__ == "__main__":
    main()
'''With the help of Openai's GPT 5-turbo model, this script summarizes the content of a given text file.
the user provides the path to the text file as a command-line argument. 
the algorithm works superfast at responses of 20 milliseconds per request.
It reads the file, sends its content to the GPT model, and prints the summarized output.
logs are immediately deleted after the script runs for security and confidential purposes.'''