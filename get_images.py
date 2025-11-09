import requests
import time
from datetime import datetime
import os

PI_HOST = "10.194.194.1"
PHOTO_FILENAME = "test.jpg"  # Name of photo on Pi's desktop
LOCAL_SAVE_DIR = ".\\test_images\\"

os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)

try:
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"{LOCAL_SAVE_DIR}photo_{timestamp}.jpg"
        
        response = requests.get(f"http://{PI_HOST}:8080/{PHOTO_FILENAME}")
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded: {local_filename}")
        time.sleep(60)
        
except KeyboardInterrupt:
    print("\nStopping...")
