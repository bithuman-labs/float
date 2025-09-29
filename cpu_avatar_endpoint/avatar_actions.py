#!/usr/bin/env python3
"""
Avatar Action Controls

Provides a clean interface for triggering avatar animations and expressions.
Uses the official bithuman VideoControl class for compatibility.
Includes debounce mechanism to prevent duplicate gesture triggers.
"""

import logging
import time
from functools import wraps
from typing import Any, Dict, Optional, List, Union, Callable, Awaitable

logger = logging.getLogger("avatar-actions")

# Global gesture cooldown tracking to prevent duplicate triggers
_gesture_cooldowns: Dict[str, float] = {}

# Message deduplication tracking to prevent broadcast duplicates
_processed_messages: Dict[str, float] = {}

# Default cooldown times for different gesture types
DEFAULT_COOLDOWNS = {
    "mini_wave_hello": 3.0,
    "nod": 2.0,
    "smile": 1.5,
    "default": 2.0
}

try:
    # Import the official VideoControl from bithuman package
    from bithuman.api import VideoControl
    logger.info("Successfully imported VideoControl from bithuman.api")
except ImportError as e:
    logger.warning(f"Could not import VideoControl from bithuman.api: {e}")
    # Fallback to a local implementation if bithuman is not available
    from dataclasses import dataclass, field
    import uuid
    
    @dataclass
    class VideoControl:
        """
        Fallback VideoControl implementation with target_video attribute
        Compatible with bithuman VideoControl API
        """
        action: Optional[Union[str, List[str]]] = None
        target_video: Optional[str] = None
        text: Optional[str] = None
        message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
        end_of_speech: bool = False
        
        # Additional parameters for convenience
        parameters: Optional[Dict[str, Any]] = None
        duration: Optional[float] = None
        intensity: Optional[float] = None
        
        def __post_init__(self):
            """Validate the action and parameters"""
            if self.action:
                valid_actions = [
                    "mini_wave_hello",
                    "wave", 
                    "nod",
                    "shake_head",
                    "smile", 
                    "blink",
                    "look_left",
                    "look_right",
                    "idle"
                ]
                
                actions_to_check = [self.action] if isinstance(self.action, str) else self.action
                for action in actions_to_check:
                    if action not in valid_actions:
                        logger.warning(f"Unknown action: {action}. Valid actions: {valid_actions}")
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary for logging/debugging"""
            return {
                "action": self.action,
                "target_video": self.target_video,
                "text": self.text,
                "message_id": self.message_id,
                "end_of_speech": self.end_of_speech,
                "parameters": self.parameters,
                "duration": self.duration,
                "intensity": self.intensity
            }


def _check_cooldown(gesture_name: str, cooldown: float) -> bool:
    """
    Check if a gesture is still in cooldown period
    
    Args:
        gesture_name: Name of the gesture
        cooldown: Cooldown time in seconds
        
    Returns:
        bool: True if available (not in cooldown), False if still cooling down
    """
    current_time = time.time()
    last_trigger = _gesture_cooldowns.get(gesture_name, 0)
    
    if current_time - last_trigger < cooldown:
        remaining_cooldown = cooldown - (current_time - last_trigger)
        logger.info(f"Gesture '{gesture_name}' still in cooldown ({remaining_cooldown:.1f}s remaining)")
        return False
    
    return True


def _update_cooldown(gesture_name: str) -> None:
    """Update the cooldown timestamp for a gesture"""
    _gesture_cooldowns[gesture_name] = time.time()


async def execute_video_control(runtime: Any, control: VideoControl) -> bool:
    """
    Execute a video control command on the runtime
    
    Args:
        runtime: AsyncBithuman runtime instance
        control: VideoControl command to execute
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Executing video control: {control.action}")
        
        # Check if runtime has the push method
        if not hasattr(runtime, 'push'):
            logger.error("Runtime does not have push method")
            return False
        
        await runtime.push(control)
        
        logger.info(f"Video control '{control.action}' executed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to execute video control '{control.action}': {str(e)}")
        return False


def _check_message_duplicate(message_id: str, dedup_window: float = 10.0) -> bool:
    """
    Check if a message has been processed recently (for broadcast deduplication)
    
    Args:
        message_id: Unique identifier for the message
        dedup_window: Time window in seconds to consider duplicates
        
    Returns:
        bool: True if duplicate (should skip), False if new message
    """
    current_time = time.time()
    
    # Clean old entries
    expired_keys = [k for k, v in _processed_messages.items() if current_time - v > dedup_window]
    for key in expired_keys:
        del _processed_messages[key]
    
    # Check if message was already processed
    if message_id in _processed_messages:
        last_processed = _processed_messages[message_id]
        if current_time - last_processed < dedup_window:
            logger.info(f"Duplicate message detected: {message_id} (processed {current_time - last_processed:.1f}s ago)")
            return True
    
    # Mark message as processed
    _processed_messages[message_id] = current_time
    return False


async def trigger_gesture_with_debounce(runtime: Any, action: str, target_video: Optional[str] = None, 
                                       cooldown: Optional[float] = None, message_id: Optional[str] = None) -> bool:
    """
    Universal gesture trigger with debounce protection and message deduplication
    
    Args:
        runtime: AsyncBithuman runtime instance
        action: The gesture action name
        target_video: Optional target video parameter
        cooldown: Custom cooldown time, uses default if None
        message_id: Optional message ID for broadcast deduplication
        
    Returns:
        bool: True if gesture was triggered, False if still in cooldown, duplicate, or failed
    """
    # Check for message duplicates (broadcast protection)
    if message_id and _check_message_duplicate(message_id):
        logger.info(f"🔄 Skipping duplicate broadcast message for gesture: {action}")
        return False
    
    # Use default cooldown if not specified
    if cooldown is None:
        cooldown = DEFAULT_COOLDOWNS.get(action, DEFAULT_COOLDOWNS["default"])
    
    # Check cooldown
    if not _check_cooldown(action, cooldown):
        return False
    
    # Execute the gesture
    control = VideoControl(action=action, target_video=target_video)
    success = await execute_video_control(runtime, control)
    
    # Update cooldown timestamp if successful
    if success:
        _update_cooldown(action)
        logger.info(f"✅ Triggered gesture: {action} (cooldown: {cooldown}s)")
    
    return success


# Convenience functions for common actions (now using the universal trigger)
async def trigger_wave_hello(runtime: Any, cooldown: Optional[float] = None, message_id: Optional[str] = None) -> bool:
    """Trigger a mini wave hello animation with debounce protection"""
    return await trigger_gesture_with_debounce(runtime, "mini_wave_hello", cooldown=cooldown, message_id=message_id)


async def trigger_nod(runtime: Any, cooldown: Optional[float] = None, message_id: Optional[str] = None) -> bool:
    """Trigger a nod animation with debounce protection"""
    return await trigger_gesture_with_debounce(runtime, "nod", cooldown=cooldown, message_id=message_id)


async def trigger_smile(runtime: Any, cooldown: Optional[float] = None, message_id: Optional[str] = None) -> bool:
    """Trigger a smile expression with debounce protection"""
    return await trigger_gesture_with_debounce(runtime, "smile", cooldown=cooldown, message_id=message_id)


async def trigger_custom_action(runtime: Any, action: str, target_video: Optional[str] = None, 
                               cooldown: Optional[float] = None, message_id: Optional[str] = None) -> bool:
    """Trigger a custom action with optional target video and debounce protection"""
    return await trigger_gesture_with_debounce(runtime, action, target_video=target_video, cooldown=cooldown, message_id=message_id)


def clear_gesture_cooldowns() -> None:
    """Clear all gesture cooldowns - useful for testing or reset scenarios"""
    global _gesture_cooldowns
    _gesture_cooldowns.clear()
    logger.info("🔄 All gesture cooldowns cleared")


def clear_message_cache() -> None:
    """Clear message deduplication cache - useful for testing or reset scenarios"""
    global _processed_messages
    _processed_messages.clear()
    logger.info("🔄 Message deduplication cache cleared")


def clear_all_caches() -> None:
    """Clear both gesture cooldowns and message cache"""
    clear_gesture_cooldowns()
    clear_message_cache()
    logger.info("🔄 All caches cleared")


def get_gesture_cooldown_status(gesture_name: str) -> Dict[str, Union[bool, float]]:
    """
    Get the cooldown status for a specific gesture
    
    Args:
        gesture_name: Name of the gesture to check
        
    Returns:
        Dict containing cooldown status information
    """
    current_time = time.time()
    last_trigger = _gesture_cooldowns.get(gesture_name, 0)
    time_since_last = current_time - last_trigger
    
    return {
        "gesture_name": gesture_name,
        "last_triggered": last_trigger,
        "time_since_last": time_since_last,
        "is_available": last_trigger == 0 or time_since_last > 0,  # Always available if no default cooldown
        "last_trigger_timestamp": last_trigger if last_trigger > 0 else None
    }
