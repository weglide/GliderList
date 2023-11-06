import requests
import json

OGN_DDB_URL = "https://ddb.glidernet.org/download/?j=1&t=1"

# aircrafts which we don not want to track
no_mapping = [
    "Scottish Aviation Bulldog",
    "A22 Foxbat",
    " FK 9 Mark VI",
    "Piccolo B",
    "Smyk",
    "Std. Austria SH1",
    "Bristell B23",
    "DJI Agras",
    "Archaeopteryx-E",
    "Thruster",
    "ICP Savannah",
    "Douglas Dakota",
    "TB-30 Epsilon",
    "Avro Lancaster",
    "AS 365",
    "PA34 Seneca",
    "DJI Mavic",
    "LET LF-109 Pionyr",
    "Bell 429",
    "RF 47",
    "Swiftlight",
    "Viper SD-4",
    "AFH-22",
    "AK-8",
    "Parrot AR.Drone",
    "Slingsby T-21B",
    "SZD-45 Ogar",
    "Aeromot AMT-200S Super Ximango",
    "Swiftlight-E",
    "DJI Matrice",
    "Archaeopteryx",
    "Piper PA-44",
    "Piper PA-38 Tomahawk",
    "Slingsby T.43 Skylark 3C",
    "Schreder HP-18",
    "Zuni II",
    "Apollo Fox",
    "C 38",
    "Vans RV-6",
    "DJI Mini",
    "Parachute",
    "Glasair III",
    "Cessna 404",
    "PZL W-3 Sokol",
    "Bristell NG5",
    "SGS 2-32",
    "Slingsby T34 Sky",
    "MS-887 Rallye 125",
    "Mu 28",
    "HP-24",
    "Helisports CH77",
    "SZD-22 Mucha Standard",
    "Piper PA-17",
    "LP-15 Nugget",
    "Glastar",
]

mapping_file = open("ogn_ddb_mapping.json")
mapping = json.load(mapping_file)
all_we_know = list(mapping.keys()) + no_mapping


def test_all_mapped():
    """Test if all aircraft names in the ogn ddb are either mapped to a WeGlide type or ignored"""
    response = requests.get(OGN_DDB_URL)
    devices = response.json()["devices"]
    for device in devices:
        name = device["aircraft_model"]
        if not name:
            continue

        assert name in all_we_know

