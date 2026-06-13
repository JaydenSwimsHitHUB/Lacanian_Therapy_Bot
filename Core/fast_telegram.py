import asyncio
import time
import logging
from rasa.core.channels.telegram import TelegramInput

# Establish a direct line to the screen so we can see the logs
logger = logging.getLogger(__name__)

class FastTelegramInput(TelegramInput):
    @classmethod
    def name(cls) -> str:
        return "fast_telegram"

    def blueprint(self, on_new_message):
        
        async def fast_on_new_message(user_message):
            trigger_time = time.time()
            message_id = id(user_message)
            
            # --- START OF OUR ALARM SYSTEM ---
            # 1. Grab any extra data attached to the message, safely handling if there is none
            metadata = getattr(user_message, 'metadata', {}) or {}
            
            # 2. Find out the exact moment the user hit send on their phone
            telegram_sent_time = metadata.get("timestamp", trigger_time)
            
            # 3. Calculate how long the message spent traveling through the internet
            network_transit_time = trigger_time - telegram_sent_time
            
            # 4. Sound the alarm if it took longer than 5 seconds to reach our front door
            if network_transit_time > 5.0:
                logger.warning(f"[ALARM] Network Bottleneck Detected! Message took {network_transit_time:.2f}s to reach us.")
            # --- END OF OUR ALARM SYSTEM ---
            
            # Log that the receptionist took the message
            logger.warning(f"[TRACKER-{message_id}] 1. Webhook triggered. Transit time: {network_transit_time:.2f}s")
            
            async def safe_process():
                try:
                    logger.warning(f"[TRACKER-{message_id}] 2. Entering Rasa pipeline.")
                    start = time.time()
                    
                    # Hand the message over to the bot's brain
                    await on_new_message(user_message)
                    
                    end = time.time()
                    logger.warning(f"[TRACKER-{message_id}] 3. Pipeline complete in {end - start:.2f}s.")
                except Exception as e:
                    logger.error(f"[TRACKER-{message_id}] ERROR: Pipeline crashed: {e}")

            # Send the actual thinking process to the background so the receptionist is free again
            asyncio.create_task(safe_process())
            
            # A tiny pause to let the background task start smoothly
            await asyncio.sleep(0) 
            
            finish_time = time.time()
            logger.warning(f"[TRACKER-{message_id}] 4. Handed back to server in {finish_time - trigger_time:.4f}s.")

        return super().blueprint(fast_on_new_message)