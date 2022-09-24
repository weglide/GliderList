import requests
import json

ogn_ddb_url = "https://ddb.glidernet.org/download/?j=1&t=1"


def download_ddb():
    ddb = requests.get(ogn_ddb_url).json()
    aircraft = {v: {"id": 1, "name": v} for v in set([v["aircraft_model"] for v in ddb["devices"]])}
    
    with open("ogn_ddb.json", "w") as outfile:
        json.dump(ddb, outfile, indent=4)

    with open("aircraft.json", "w") as outfile:
        json.dump(aircraft, outfile, indent=4)


if __name__ == "__main__":
    download_ddb()
    