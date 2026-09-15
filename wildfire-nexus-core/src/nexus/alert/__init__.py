"""
Alert module for notifications.
"""

from .telegram_bot import send_telegram_alert

__all__ = ["send_telegram_alert"]
