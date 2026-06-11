import asyncio
import time
import logging
from rasa.core.channels.telegram import TelegramInput

logger = logging.getLogger(__name__)

class FastTelegramInput(TelegramInput):
    @classmethod
    def name(cls) -> str:
        return "fast_telegram"

    def blueprint(self, on_new_message):
        
        # 1. Background worker to prevent Telegram webhook duplication
        async def fast_on_new_message(user_message):
            async def safe_process():
                try:
                    await on_new_message(user_message)
                except Exception as e:
                    logger.error(f"[ALARM] ERROR: Pipeline crashed: {e}", exc_info=True)

            # Dispatch task and yield immediately to return 200 OK
            asyncio.create_task(safe_process())
            await asyncio.sleep(0)

        # 2. Generate standard routing blueprint
        bp = super().blueprint(fast_on_new_message)

        # 3. Intercept payload for timestamp analysis
        @bp.middleware('request')
        async def intercept_telegram_timestamp(request):
            if request.method == "POST":
                try:
                    # Safely access JSON; Sanic returns None if content-type is wrong
                    data = request.json
                    if not data:
                        return
                    
                    message_data = data.get("message") or data.get("edited_message")
                    
                    if message_data and "date" in message_data:
                        # Telegram timestamp is an integer (seconds since epoch)
                        telegram_sent_time = int(message_data["date"])
                        current_time = time.time()
                        
                        network_transit_time = current_time - telegram_sent_time
                        
                        # Observation: Log the exact transit metric for every message
                        logger.info(
                            f"[METRIC] Transit Time: {network_transit_time:.2f}s | "
                            f"Telegram Time: {telegram_sent_time} | Server Time: {current_time:.2f}"
                        )
                        
                except (AttributeError, KeyError, TypeError):
                    # Ignore malformed payloads that lack expected dictionary structures
                    pass
                except Exception as e:
                    # Log unexpected exceptions instead of swallowing them
                    logger.error(f"Failed to inspect Telegram payload: {e}")
        
        return bp