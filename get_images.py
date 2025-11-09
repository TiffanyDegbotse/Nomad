import subprocess
import time
from datetime import datetime

# Configuration
PI_USER = "mrinalgoel"
PI_PASSWORD = "pi123"
PI_HOST = "10.194.194.1"
REMOTE_PHOTO_PATH = "/home/mrinalgoel/Desktop/test3.jpg"
LOCAL_SAVE_DIR = "./test_photos/"

# Ensure local directory exists
subprocess.run(f"mkdir -p {LOCAL_SAVE_DIR}", shell=True)

try:
    while True:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_filename = f"{LOCAL_SAVE_DIR}photo_{timestamp}.jpg"
        
        scp_cmd = f"sshpass -p '{PI_PASSWORD}' scp {PI_USER}@{PI_HOST}:{REMOTE_PHOTO_PATH} {local_filename}"
        
        # Download to local machine
        subprocess.run(scp_cmd, shell=True)
        
        print(f"Downloaded: {local_filename}")
        
        # Wait 1 minute
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\nStopping...")
