import os
from typing import NamedTuple

import requests
from dotenv import load_dotenv


class Environment(NamedTuple):
    """Environment variables to execute a command against the WeGlide backend."""

    user_email: str
    user_password: str
    client_id: str
    api_key: str
    api_url: str

    @classmethod
    def from_env(cls):
        load_dotenv()
        try:
            return cls(
                user_email=os.environ["USER_EMAIL"],
                user_password=os.environ["USER_PASSWORD"],
                client_id=os.environ["CLIENT_ID"],
                api_key=os.environ["API_KEY"],
                api_url=os.environ["API_URL"],
            )
        except KeyError as e:
            raise Exception(f"Missing environment variable: {e}")


class ApiKeyCommand:
    """
    Trigger a command on the WeGlide backend.
    """

    def __init__(self):
        self.env = Environment.from_env()

    @property
    def access_token(self) -> str:
        """Get an access token for the WeGlide backend."""
        form_data = {
            "username": self.env.user_email,
            "password": self.env.user_password,
            "client_id": self.env.client_id,
            "grant_type": "password",
        }
        url = f"{self.env.api_url}auth/token"
        response = requests.post(url, data=form_data)

        # We are interested in the full response in case of an error, so we don't rely on `response.raise_for_status()`.
        if response.status_code != 200:
            raise Exception(
                f"Failed to get access token: {response.status_code}, {response.text}"
            )

        return response.json()["access_token"]

    def post_command(self):
        """Trigger a command on the WeGlide backend."""
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "X-API-KEY": self.env.api_key,
        }
        url = f"{self.env.api_url}admin/api-key-command"
        response = requests.post(
            url,
            json={"command": "load_aircraft", "kwargs": {}},
            headers=headers,
        )

        # We are interested in the full response in case of an error, so we don't rely on `response.raise_for_status()`.
        if response.status_code != 200:
            raise Exception(
                f"Command failed with status code {response.status_code}, {response.text}"
            )
        print("Command succeeded with response:", response.text)


if __name__ == "__main__":
    ApiKeyCommand().post_command()