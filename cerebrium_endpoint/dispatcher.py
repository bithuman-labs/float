import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from livekit.agents.cli.log import setup_logging

setup_logging("INFO", devmode=True, console=False)

logger = logging.getLogger("avatar-dispatcher")

THIS_DIR = Path(__file__).parent.absolute()

FLOAT_CHECKPOINT_PATH = os.getenv(
    "FLOAT_CHECKPOINT_PATH", "/persistent-storage/float-checkpoints"
)


@dataclass
class AvatarConnectionInfo:
    room_name: str
    url: str
    token: str
    avatar_id: Optional[str] = None
    avatar_image_filename: Optional[str] = None


class WorkerLauncher:
    """Local implementation that launches workers as subprocesses"""

    @dataclass
    class WorkerInfo:
        room_name: str
        process: subprocess.Popen
        done_fut: asyncio.Future[None]

    def __init__(self):
        self.workers: dict[str, WorkerLauncher.WorkerInfo] = {}
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._monitor_task = asyncio.create_task(self._monitor())

    def close(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()

        for worker in self.workers.values():
            worker.process.terminate()
            try:
                worker.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.process.kill()

    async def launch_worker(self, connection_info: AvatarConnectionInfo) -> WorkerInfo:
        if connection_info.room_name in self.workers:
            worker = self.workers[connection_info.room_name]
            worker.process.terminate()
            try:
                worker.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.process.kill()

        # Launch new worker process
        env = os.environ.copy()
        env["LIVEKIT_ROOM"] = connection_info.room_name
        env["LIVEKIT_WS_URL"] = connection_info.url
        env["LIVEKIT_TOKEN"] = connection_info.token
        env["FLOAT_CHECKPOINT_PATH"] = FLOAT_CHECKPOINT_PATH
        if connection_info.avatar_image_filename:
            env["REF_IMAGE_PATH"] = connection_info.avatar_image_filename

        cmd = [
            sys.executable,
            str(THIS_DIR / "avatar_worker.py"),
        ]

        try:
            room_name = connection_info.room_name
            process = subprocess.Popen(
                cmd, env=env, stdout=sys.stdout, stderr=sys.stderr
            )
            self.workers[room_name] = WorkerLauncher.WorkerInfo(
                room_name=room_name, process=process, done_fut=asyncio.Future()
            )
            return self.workers[room_name]
        except Exception as e:
            logger.exception(f"Failed to launch worker: {e}")
            raise HTTPException(status_code=500, detail=str(e))  # noqa: B904

    async def _monitor(self) -> None:
        while True:
            for worker in list(self.workers.values()):
                if worker.process.poll() is not None:
                    logger.info(
                        f"Worker for room {worker.room_name} exited with code {worker.process.returncode}"  # noqa: E501
                    )
                    self.workers.pop(worker.room_name)
                    worker.done_fut.set_result(None)
            await asyncio.sleep(1)


launcher = WorkerLauncher()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await launcher.start()
    yield
    launcher.close()


app = FastAPI(title="Avatar Dispatcher", lifespan=lifespan)


@app.post("/launch")
async def handle_launch(
    livekit_url: str = Form(...),
    livekit_token: str = Form(...),
    room_name: str = Form(...),
    avatar_id: Optional[str] = Form(None),
    avatar_image: Optional[UploadFile] = File(None),
) -> dict:
    """Handle request to launch an avatar worker"""

    avatar_image_filename: str | None = None

    # Process avatar image if provided - write to temp file
    if avatar_image:
        try:
            # Use the uploaded filename directly
            filename = os.path.basename(avatar_image.filename) or "avatar.jpg"
            avatar_image_filename = tempfile.NamedTemporaryFile(
                delete=False, prefix=f"{room_name}_", suffix=f"_{filename}"
            ).name

            # Write image data to temp file
            image_data = await avatar_image.read()
            with open(avatar_image_filename, "wb") as f:
                f.write(image_data)

            logger.info(
                f"Saved avatar image to: {avatar_image_filename}, size: {len(image_data)} bytes"
            )

        except Exception as e:
            logger.error(f"Failed to save avatar image: {e}")
            # Continue without image if there's an error
            avatar_image_filename = None

    connection_info = AvatarConnectionInfo(
        room_name=room_name,
        url=livekit_url,
        token=livekit_token,
        avatar_id=avatar_id,
        avatar_image_filename=avatar_image_filename,
    )

    try:
        worker = await launcher.launch_worker(connection_info)
        logger.info(f"Launched avatar worker for room: {connection_info.room_name}")
        tic = time.time()
        await worker.done_fut
        toc = time.time()
        logger.info(
            f"[{connection_info.room_name}] Avatar worker exited after {toc - tic:.2f} seconds"
        )
        return {
            "status": "success",
            "message": f"Avatar worker launched for room: {connection_info.room_name}",
            "duration": toc - tic,
        }
    except Exception as e:
        logger.error(f"Error handling launch request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to launch worker: {str(e)}"
        )  # noqa: B904
    finally:
        if avatar_image_filename and os.path.isfile(avatar_image_filename):
            os.unlink(avatar_image_filename)
            logger.info(f"Cleaned up temp avatar image: {avatar_image_filename}")


@app.get("/health")
async def handle_health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
async def handle_ready() -> dict:
    return {"status": "ok"}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default="0.0.0.0", type=str, help="Host to run server on"
    )
    parser.add_argument("--port", default=8089, type=int, help="Port to run server on")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
