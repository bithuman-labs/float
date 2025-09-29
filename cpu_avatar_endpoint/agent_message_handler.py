#!/usr/bin/env python3
"""
Agent Message Handler for Avatar Service

Handles incoming greeting messages.
Triggers avatar wave animation when greetings are received.
"""

import json
import logging
from datetime import datetime
from typing import Any

from avatar_actions import trigger_wave_hello
from livekit import rtc

logger = logging.getLogger("agent-message-handler")


async def setup_agent_message_handlers(room: rtc.Room, runtime: Any) -> bool:
    """
    Setup greeting message handler

    Args:
        room: LiveKit room instance
        runtime: AsyncBithuman runtime instance for triggering animations

    Returns:
        bool: True if setup successful, False otherwise
    """
    try:
        local_participant = room.local_participant

        # Handler for greeting messages from agents
        @local_participant.register_rpc_method("handle_greeting")
        async def handle_greeting(data: rtc.RpcInvocationData) -> str:
            """Handle greeting messages and trigger wave animation"""
            try:
                logger.info(
                    f"Received greeting from {data.caller_identity}: {data.payload}"
                )

                # Parse the greeting data
                payload = json.loads(data.payload) if data.payload else {}
                greeting_data = payload.get("data", {})

                # Extract key information
                agent_code = greeting_data.get("agent_code", "unknown")
                user_id = greeting_data.get("user_id", "unknown")
                message = greeting_data.get("message", "")
                model = greeting_data.get("model", "essence")

                logger.info(
                    f"Processing greeting from agent {agent_code} (user: {user_id}, model: {model})"
                )
                logger.info(f"Message: {message[:100]}...")

                # Trigger wave animation if runtime is available (with enhanced debounce protection)
                animation_success = False
                animation_debounced = False
                if runtime:
                    try:
                        # Extract greeting timestamp and create message ID for duplicate detection
                        greeting_timestamp = greeting_data.get("timestamp", 0)
                        # Use deduplication key from core-agent-worker if available, otherwise create one
                        message_id = (
                            greeting_data.get("message_dedup_key")
                            or f"{agent_code}:{user_id}:{greeting_timestamp}"
                        )

                        # Use enhanced cooldown (5.0s) and message deduplication to prevent broadcast spam
                        # This protects against multiple avatar workers receiving the same broadcast
                        animation_success = await trigger_wave_hello(
                            runtime, cooldown=5.0, message_id=message_id
                        )
                        if animation_success:
                            logger.info(
                                f"🎭 Successfully triggered mini wave hello animation (timestamp: {greeting_timestamp})"
                            )
                        else:
                            # Check if it was debounced or actually failed
                            from avatar_actions import get_gesture_cooldown_status

                            cooldown_status = get_gesture_cooldown_status(
                                "mini_wave_hello"
                            )
                            if cooldown_status["time_since_last"] < 5.0:
                                animation_debounced = True
                                logger.info(
                                    f"🚫 Wave animation debounced - last triggered {cooldown_status['time_since_last']:.1f}s ago (broadcast protection)"
                                )
                            else:
                                logger.warning(
                                    "❌ Failed to trigger wave animation - runtime error"
                                )
                    except Exception as e:
                        logger.error(f"💥 Error triggering animation: {str(e)}")
                else:
                    logger.warning("⚠️ Runtime not available - skipping animation")

                # Log analytics with enhanced debounce information
                analytics_data = {
                    "event_type": "avatar_greeting_received",
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_identity": data.caller_identity,
                    "agent_code": agent_code,
                    "user_id": user_id,
                    "model": model,
                    "greeting_timestamp": greeting_timestamp,
                    "message_id": message_id,
                    "animation_triggered": animation_success,
                    "animation_debounced": animation_debounced,
                    "debounce_protection": "enhanced",
                    "broadcast_protection": "enabled",
                }
                logger.info(f"[ANALYTICS] {json.dumps(analytics_data)}")

                # Return success response with debounce information
                response = {
                    "status": "success",
                    "message": "Greeting processed successfully",
                    "animation_triggered": animation_success,
                    "animation_debounced": animation_debounced,
                    "debounce_protection": "enabled",
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": "avatar_worker",
                }

                return json.dumps(response)

            except Exception as e:
                logger.error(f"Error handling greeting: {str(e)}", exc_info=True)
                error_response = {
                    "status": "error",
                    "message": f"Failed to process greeting: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                return json.dumps(error_response)

        logger.info("Agent message handlers registered successfully")
        logger.info("Registered method: handle_greeting")

        return True

    except Exception as e:
        logger.error(f"Failed to setup RPC handlers: {str(e)}", exc_info=True)
        return False
