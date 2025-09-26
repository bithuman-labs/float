#!/usr/bin/env python3
"""
Avatar Action Controls

Provides a clean interface for triggering avatar animations and expressions.
Uses the official bithuman VideoControl class for compatibility.
"""

import logging
from typing import Any, Dict, Optional, List, Union

logger = logging.getLogger("avatar-actions")

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


# Convenience functions for common actions
async def trigger_wave_hello(runtime: Any) -> bool:
    """Trigger a mini wave hello animation"""
    control = VideoControl(action="mini_wave_hello", force_action=True)
    return await execute_video_control(runtime, control)


async def trigger_nod(runtime: Any) -> bool:
    """Trigger a nod animation"""
    control = VideoControl(action="nod")
    return await execute_video_control(runtime, control)


async def trigger_smile(runtime: Any) -> bool:
    """Trigger a smile expression"""
    control = VideoControl(action="smile")
    return await execute_video_control(runtime, control)


async def trigger_custom_action(runtime: Any, action: str, target_video: Optional[str] = None) -> bool:
    """Trigger a custom action with optional target video"""
    control = VideoControl(action=action, target_video=target_video)
    return await execute_video_control(runtime, control)
