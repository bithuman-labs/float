#!/usr/bin/env python3
"""
Agent Message Handler for Avatar Service

Handles incoming greeting messages.
Triggers avatar wave animation when greetings are received.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict

from livekit import rtc
from avatar_actions import trigger_wave_hello

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
                logger.info(f"Received greeting from {data.caller_identity}: {data.payload}")
                
                # Parse the greeting data
                payload = json.loads(data.payload) if data.payload else {}
                greeting_data = payload.get("data", {})
                
                # Extract key information
                agent_code = greeting_data.get("agent_code", "unknown")
                user_id = greeting_data.get("user_id", "unknown")
                message = greeting_data.get("message", "")
                model = greeting_data.get("model", "essence")
                
                logger.info(f"Processing greeting from agent {agent_code} (user: {user_id}, model: {model})")
                logger.info(f"Message: {message[:100]}...")
                
                # Trigger wave animation if runtime is available
                animation_success = False
                if runtime:
                    try:
                        animation_success = await trigger_wave_hello(runtime)
                        if animation_success:
                            logger.info("Successfully triggered mini wave hello animation")
                        else:
                            logger.warning("Failed to trigger wave animation")
                    except Exception as e:
                        logger.error(f"Error triggering animation: {str(e)}")
                else:
                    logger.warning("Runtime not available - skipping animation")
                
                # Log analytics
                analytics_data = {
                    "event_type": "avatar_greeting_received",
                    "timestamp": datetime.utcnow().isoformat(),
                    "agent_identity": data.caller_identity,
                    "agent_code": agent_code,
                    "user_id": user_id,
                    "model": model,
                    "animation_triggered": animation_success
                }
                logger.info(f"[ANALYTICS] {json.dumps(analytics_data)}")
                
                # Return success response
                response = {
                    "status": "success",
                    "message": "Greeting processed successfully",
                    "animation_triggered": animation_success,
                    "timestamp": datetime.utcnow().isoformat(),
                    "service": "avatar_worker"
                }
                
                return json.dumps(response)
                
            except Exception as e:
                logger.error(f"Error handling greeting: {str(e)}", exc_info=True)
                error_response = {
                    "status": "error",
                    "message": f"Failed to process greeting: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                return json.dumps(error_response)
        
        logger.info("Agent message handlers registered successfully")
        logger.info("Registered method: handle_greeting")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup RPC handlers: {str(e)}", exc_info=True)
        return False
