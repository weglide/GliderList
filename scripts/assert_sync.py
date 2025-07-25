import os

import requests
from dotenv import load_dotenv


def latest_aircraft_id() -> str:
    with open("gliderlist.csv", "r") as f:
        last_line = f.readlines()[-1]
        return last_line.split(",")[0]


def last_aircraft_synced():
    url = os.environ["API_URL"]
    last_id = latest_aircraft_id()
    url = f"{url}aircraft/{last_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(
            f"Failed to get aircraft {last_id}: {response.status_code}, {response.text}"
        )

    print(f"Aircraft {last_id} exists.")


if __name__ == "__main__":
    load_dotenv()
    last_aircraft_synced()
