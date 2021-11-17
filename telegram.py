"""Telegram bot which sends a string to a user specified in credentials.py

Reference:
https://www.geeksforgeeks.org/send-message-to-telegram-user-using-python/

Author: Maria Gospodinova
"""

from telethon.sync import TelegramClient

import credentials


class TelegramBot():
    """Class with method to send message to Telegram bot."""

    def __init__(self) -> None:
        self.api_id = credentials.TELEGRAM_API_ID
        self.api_hash = credentials.TELEGRAM_API_HASH
        self.phone = credentials.PHONE_NUMBER

    def send_telegram(self, message: str) -> None:
        """Send message to Telegram bot.

        Arguments:
            message {str} -- string to send to Telegram bot
        """

        # Create telegram session and assign to variable client
        client = TelegramClient("session", self.api_id, self.api_hash)

        # Connect and build session
        client.connect()

        # If authorisation unsuccessful, send one-time-passcode to phone number
        if not client.is_user_authorized():
            client.send_code_request(self.phone)
            client.sign_in(self.phone, input("Enter the code: "))

        try:
            client.send_message("me", message, parse_mode="html")
        except Exception as e:
            print(e)

        client.disconnect()
