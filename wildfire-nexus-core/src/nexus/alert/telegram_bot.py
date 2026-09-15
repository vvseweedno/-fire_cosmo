"""
Telegram bot for fire alerts.

Sends notifications to configured Telegram chat.
"""

import asyncio
import logging

from aiogram import Bot, exceptions

from ..db.models import FireEvent
from ..settings import get_settings

logger = logging.getLogger(__name__)


def _format_alert_message(event: FireEvent, alert_type: str = "NEW") -> str:
    """
    Format fire event into Telegram message.
    
    Args:
        event: FireEvent instance
        alert_type: NEW or UPDATE
        
    Returns:
        Formatted message string
    """
    # Emoji based on risk level
    emojis = {
        "critical": "🔴",
        "warning": "🟠",
        "info": "🔵",
    }
    emoji = emojis.get(event.risk_level, "🔵")
    
    # Build message
    lines = [
        f"{emoji} {event.risk_level.upper()} | Пожар #{event.id[:6]}",
        f"📍 {event.centroid_lat:.4f}, {event.centroid_lon:.4f}",
    ]
    
    if event.nearest_settlement:
        lines.append(f"🏘 {event.nearest_settlement}")
    
    if event.area_estimate_ha and event.area_estimate_ha > 0:
        lines.append(f"📏 Площадь ~{event.area_estimate_ha:.0f} га")
    
    lines.append(f"🔥 Точек зафиксировано: {event.point_count}")
    
    if event.max_frp and event.max_frp > 0:
        lines.append(f"⚡ Max FRP: {event.max_frp:.1f} MW")
    
    lines.append(f"⏰ Обнаружен: {event.first_seen.strftime('%Y-%m-%d %H:%M UTC')}")
    
    # Add map link
    map_url = f"http://localhost:8000/#event={event.id}"
    lines.append(f"🗺 Карта: {map_url}")
    
    if alert_type == "UPDATE":
        lines.append("\n⚠️ ОБНОВЛЕНИЕ статуса")
    
    return "\n".join(lines)


async def send_telegram_alert(event: FireEvent, alert_type: str = "NEW") -> bool:
    """
    Send fire alert to Telegram.
    
    Args:
        event: FireEvent to alert about
        alert_type: NEW or UPDATE
        
    Returns:
        True if sent successfully, False otherwise
        
    Note:
        This is non-blocking - failures are logged but don't stop execution
    """
    settings = get_settings()
    
    # Check if Telegram is configured
    if not settings.telegram_bot_token:
        logger.debug("Telegram not configured (no token), skipping alert")
        return False
    
    if not settings.telegram_chat_id:
        logger.debug("Telegram chat ID not configured, skipping alert")
        return False
    
    # Format message
    message = _format_alert_message(event, alert_type)
    
    try:
        # Create bot and send message
        bot = Bot(token=settings.telegram_bot_token)
        
        await bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=message,
            parse_mode="HTML",
        )
        
        await bot.session.close()
        
        logger.info(f"Telegram alert sent for event {event.id}")
        return True
        
    except exceptions.TelegramAPIError as e:
        logger.error(f"Telegram API error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False


def send_telegram_alert_sync(event: FireEvent, alert_type: str = "NEW") -> bool:
    """
    Synchronous wrapper for sending Telegram alerts.
    
    Args:
        event: FireEvent to alert about
        alert_type: NEW or UPDATE
        
    Returns:
        True if sent successfully, False otherwise
    """
    try:
        return asyncio.run(send_telegram_alert(event, alert_type))
    except RuntimeError:
        # Event loop already running (e.g., in async context)
        logger.warning("Event loop already running, scheduling alert asynchronously")
        asyncio.create_task(send_telegram_alert(event, alert_type))
        return True
