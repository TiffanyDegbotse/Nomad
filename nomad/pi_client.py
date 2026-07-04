#!/usr/bin/env python3
"""Poll the Raspberry Pi camera server and save images locally."""

import time
from datetime import datetime

import requests

from nomad import config

PHOTO_FILENAME = "test.jpg"


def poll_images(interval=None):
    config.ensure_dirs()
    interval = interval or config.IMAGE_POLL_INTERVAL

    print(f"📡 Polling Pi at {config.PI_HOST_IP}:8080 every {interval}s")
    print(f"💾 Saving to {config.IMAGES_STREAM_DIR}")

    try:
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_filename = config.IMAGES_STREAM_DIR / f"photo_{timestamp}.jpg"

            response = requests.get(
                f"http://{config.PI_HOST_IP}:8080/{PHOTO_FILENAME}",
                timeout=10,
            )
            local_filename.write_bytes(response.content)
            print(f"Downloaded: {local_filename}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping...")


def main():
    poll_images()


if __name__ == "__main__":
    main()
