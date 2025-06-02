import asyncio
import logging
import sys
from typing import Optional
import argparse

from livekit import rtc
from livekit.agents.voice.avatar import (
    AvatarOptions,
    AvatarRunner,
    DataStreamAudioReceiver,
)
from generate2 import FloatVideoGen, BaseOptions

logger = logging.getLogger("avatar-example")


async def main(room: rtc.Room, opt: argparse.Namespace):
    """Main application logic for the avatar worker"""
    runner: AvatarRunner | None = None
    stop_event = asyncio.Event()

    try:
        # Initialize and start worker
        avatar_options = AvatarOptions(
            video_width=512,
            video_height=512,
            video_fps=25,
            audio_sample_rate=16000,
            audio_channels=1,
        )
        video_gen = FloatVideoGen(opt)
        video_gen.start(ref_image=opt.ref_path)
        runner = AvatarRunner(
            room,
            audio_recv=DataStreamAudioReceiver(room),
            video_gen=video_gen,
            options=avatar_options,
        )
        await runner.start()

        # Set up disconnect handler
        async def handle_disconnect(participant: rtc.RemoteParticipant):
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                logging.info(
                    "Agent %s disconnected, stopping worker...", participant.identity
                )
                stop_event.set()

        room.on(
            "participant_disconnected",
            lambda p: asyncio.create_task(handle_disconnect(p)),
        )
        room.on("disconnected", lambda _: stop_event.set())

        # Wait until stopped
        await stop_event.wait()

    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise
    finally:
        if runner:
            await runner.aclose()
        video_gen.close()


async def run_service(url: str, token: str, opt: argparse.Namespace):
    """Run the avatar worker service"""
    room = rtc.Room()
    try:
        # Connect to LiveKit room
        logging.info("Connecting to %s", url)
        await room.connect(url, token)
        logging.info("Connected to room %s", room.name)

        def handle_transcription(reader: rtc.TextStreamReader, participant: rtc.RemoteParticipant):
            pass

        room.register_text_stream_handler("lk.transcription", handle_transcription)

        # Run main application logic
        await main(room, opt)
    except rtc.ConnectError as e:
        logging.error("Failed to connect to room: %s", e)
        raise
    finally:
        await room.disconnect()


class InferenceOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser: argparse.ArgumentParser):
        super().initialize(parser)
        parser.add_argument(
            "--ref_path",
            default="assets/Snipaste_2025-03-24_17-16-02.png",
            type=str,
            help="ref",
        )
        parser.add_argument(
            "--emo",
            default=None,
            type=str,
            help="emotion",
            choices=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        )
        parser.add_argument("--no_crop", action="store_true", help="not using crop")
        parser.add_argument(
            "--ckpt_path",
            default="checkpoints/float.pth",
            type=str,
            help="checkpoint path",
        )
        # LiveKit options
        parser.add_argument("--url", required=True, help="LiveKit server URL")
        parser.add_argument("--token", required=True, help="Token for joining room")
        parser.add_argument("--room", help="Room name")
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Log level",
        )
        return parser


if __name__ == "__main__":
    import sys

    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1
    opt.seed = 15
    opt.a_cfg_scale = 2.0
    opt.e_cfg_scale = 1.0

    def setup_logging(room: Optional[str], level: str):
        """Set up logging configuration"""
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        if room:
            log_format = f"[{room}] {log_format}"

        logging.basicConfig(level=getattr(logging, level.upper()), format=log_format)

    setup_logging(opt.room, opt.log_level)
    try:
        asyncio.run(run_service(opt.url, opt.token, opt))
    except KeyboardInterrupt:
        logging.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)
    finally:
        logging.info("Shutting down...")
