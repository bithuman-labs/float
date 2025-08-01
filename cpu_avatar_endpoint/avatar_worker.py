import asyncio
import logging
import os
import uuid
from typing import Dict

import aiohttp
from bithuman import AsyncBithuman
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import utils
from livekit.agents.cli.log import setup_logging
from livekit.agents.types import ATTRIBUTE_PUBLISH_ON_BEHALF
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
)
from livekit.plugins.bithuman.avatar import BithumanGenerator

setup_logging("INFO", devmode=True, console=False)

load_dotenv()


ROOM = os.getenv("LIVEKIT_ROOM")
WS_URL = os.getenv("LIVEKIT_WS_URL")
TOKEN = os.getenv("LIVEKIT_TOKEN")
BITHUMAN_SECRET = os.getenv("BITHUMAN_API_SECRET")  # TODO: use user's secret?
BITHUMAN_IMX_PATH = os.getenv("BITHUMAN_IMX_PATH")

logger = logging.getLogger(f"avatar-{ROOM}")


# -----------------------------------------------------------------------------
# Heartbeat Functions
# -----------------------------------------------------------------------------
async def send_heartbeat(transaction_id: str, api_secret: str, fingerprint: str = None, agent_id: str = None,
                        url: str = None, key: str = None) -> bool:
    """Send heartbeat event to OneOps API."""
    
    # Get API configuration for Cerebrium neon-services
    api_url = url or os.getenv("CEREBRIUM_NEON_SERVICES_URL", "https://api.aws.us-east-1.cerebrium.ai/v4/p-5398b08f/neon-services")
    auth_token = key or os.getenv("CEREBRIUM_AUTH_TOKEN")
    
    if not (api_url and auth_token):
        logger.error("Cerebrium neon-services API URL or auth token is not set")
        return False
    
    # Prepare headers
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {auth_token}",
        "api-secret": api_secret,
    }
    
    # Prepare data
    data = {
        "event": "beat",
        "id": str(uuid.uuid4()),
        "runtime": {
            "transaction_id": transaction_id,
            "fingerprint": fingerprint,
            "agent_id": agent_id,
            "mode": "cpu"
        }
    }
    
    try:
        logger.info(f"Sending heartbeat data: {data}")
        
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False)
        ) as session:
            async with session.post(
                f"{api_url}/v1/cloud-runtime/receive-events",
                headers=headers,
                json=data
            ) as response:
                response_data = await response.json()
                logger.info(f"Sent heartbeat response: {response_data}")
                
                # Check for 402 error (payment required)
                if response.status == 402:
                    logger.error("Heartbeat failed with 402 error - payment required")
                    return "402_error"
                    
                return response.status == 200
                
    except Exception as e:
        logger.error(f"Error sending heartbeat: {e}")
        return False


async def start_heartbeat_task(attributes: Dict[str, str], stop_event: asyncio.Event, 
                              runner: AvatarRunner, room: rtc.Room) -> asyncio.Task:
    """Start heartbeat task with error handling for 402 errors."""
    
    api_secret = attributes.get("api_secret")
    fingerprint = attributes.get("fingerprint")
    agent_id = attributes.get("agent_id")

    # Generate transaction ID
    transaction_id = str(uuid.uuid4())
    
    # Heartbeat interval (60 seconds = 1 minute)
    heartbeat_interval = 60
    
    logger.info(f"Starting heartbeat task with attributes: {attributes}")
    
    async def send_heartbeat_loop():
        """Send heartbeat every interval seconds"""
        try:
            while True:
                result = await send_heartbeat(
                    transaction_id=transaction_id,
                    api_secret=api_secret,
                    fingerprint=fingerprint,
                    agent_id=agent_id,
                )
                
                if result == "402_error":
                    logger.error("Heartbeat returned 402 error - stopping runner and disconnecting")
                    # Stop the runner
                    await runner.aclose()
                    # Disconnect from room
                    await room.disconnect()
                    # Set stop event to end main program
                    stop_event.set()
                    break
                elif not result:
                    logger.warning("Failed to send heartbeat")
                    
                await asyncio.sleep(heartbeat_interval)
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error(f"Heartbeat task failed: {e}")

    # Start heartbeat task
    heartbeat_task = asyncio.create_task(send_heartbeat_loop())
    logger.info("Heartbeat task started successfully")
    
    return heartbeat_task


async def cleanup_heartbeat_task(heartbeat_task: asyncio.Task):
    """Clean up heartbeat task."""
    if heartbeat_task and not heartbeat_task.done():
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("Heartbeat task cleaned up")


def text_stream_handler(reader: rtc.TextStreamReader, participant: str):
    pass


async def start_avatar(room: rtc.Room, video_gen: BithumanGenerator) -> AvatarRunner:
    """Main application logic for the avatar worker"""
    on_behalf_of = room.local_participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)
    if not on_behalf_of:
        raise ValueError("on_behalf_of is not set")

    room.register_text_stream_handler("lk.transcription", text_stream_handler)

    # Initialize and start worker
    output_width, output_height = video_gen.video_resolution
    avatar_options = AvatarOptions(
        video_width=output_width,
        video_height=output_height,
        video_fps=video_gen.video_fps,
        audio_sample_rate=video_gen.audio_sample_rate,
        audio_channels=1,
    )

    runner = AvatarRunner(
        room,
        audio_recv=DataStreamAudioReceiver(room, frame_size_ms=10),
        video_gen=video_gen,
        options=avatar_options,
    )
    await asyncio.wait_for(runner.start(), timeout=120)
    return runner


async def main():
    logger.info(f"loading video gen model from {BITHUMAN_IMX_PATH}")

    runtime = await AsyncBithuman.create(
        api_secret=BITHUMAN_SECRET,
        model_path=BITHUMAN_IMX_PATH,
    )

    try:
        await runtime.start()

        video_gen = BithumanGenerator(runtime)

        stop_event = asyncio.Event()

        room = rtc.Room()
        await room.connect(WS_URL, TOKEN)
        on_behalf_of = room.local_participant.attributes.get(ATTRIBUTE_PUBLISH_ON_BEHALF)

        runner = await start_avatar(room, video_gen)
        close_runner_task: asyncio.Task[None] | None = None
        
        logger.info(f"retrieve attributes: {room.local_participant.attributes}")
        # Start heartbeat task if user_id and agent_code are available
        heartbeat_task = await start_heartbeat_task(room.local_participant.attributes, stop_event, runner, room)

        @room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            nonlocal close_runner_task

            logger.info(f"Participant {participant.identity} disconnected")
            if participant.identity == on_behalf_of:
                logger.info("Agent disconnected, shutting down avatar worker...")
                if not close_runner_task:
                    close_runner_task = asyncio.create_task(runner.aclose())
                stop_event.set()

        @room.on("disconnected")
        def on_disconnected(reason: rtc.DisconnectReason):
            nonlocal close_runner_task

            logger.info(f"Room disconnected: {reason}")
            if not close_runner_task:
                close_runner_task = asyncio.create_task(runner.aclose())
            stop_event.set()

        # make sure the agent is connected, otherwise stop the worker
        try:
            await asyncio.wait_for(
                utils.wait_for_participant(room, identity=on_behalf_of), timeout=60
            )
        except asyncio.TimeoutError:
            logger.error("Agent not connected, shutting down avatar worker...")
            if not close_runner_task:
                close_runner_task = asyncio.create_task(runner.aclose())
            stop_event.set()
            return 1

        await stop_event.wait()
        
        # Clean up heartbeat task
        if heartbeat_task:
            await cleanup_heartbeat_task(heartbeat_task)
        
        return 0
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    return_code = asyncio.run(main())
    os._exit(return_code)
