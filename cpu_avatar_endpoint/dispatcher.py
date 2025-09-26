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
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException
from livekit.agents.cli.log import setup_logging

load_dotenv()

setup_logging("INFO", devmode=True, console=False)

logger = logging.getLogger("avatar-dispatcher")

THIS_DIR = Path(__file__).parent.absolute()

AGENT_MODEL_ROOT = os.getenv("AGENT_MODEL_ROOT", "/persistent-storage/agents")



@dataclass
class AvatarConnectionInfo:
    room_name: str
    url: str
    token: str
    avatar_id: str
    imx_path: str


class WorkerLauncher:
    """Local implementation that launches workers as subprocesses"""

    @dataclass
    class WorkerInfo:
        room_name: str
        process: subprocess.Popen
        done_fut: asyncio.Future[int]

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
        env["BITHUMAN_IMX_PATH"] = connection_info.imx_path

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
                    worker.done_fut.set_result(worker.process.returncode)
            await asyncio.sleep(1)


async def save_image_to_temp_file(
    image_data: bytes, room_name: str, filename: str, source: str
) -> str:
    """Create a temporary file and write image data to it"""
    # Create temporary file
    temp_filepath = tempfile.NamedTemporaryFile(
        delete=False, prefix=f"{room_name}_", suffix=f"_{filename}"
    ).name

    # Write image data to temp file
    with open(temp_filepath, "wb") as f:
        f.write(image_data)

    logger.info(
        f"{source} avatar image to: {temp_filepath}, size: {len(image_data)} bytes"
    )
    return temp_filepath


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
    avatar_id: str = Form(None),
) -> dict:
    """Handle request to launch an avatar worker"""
    logger.info(f"Launching avatar worker for room: {room_name}")
    try:
        # check if the imx model exists
        if not avatar_id:
            # use the first imx model in the directory
            avatar_id = "*"
        try:
            imx_model_path = list(Path(AGENT_MODEL_ROOT).glob(f"{avatar_id}/*.imx"))[0]
        except IndexError:
            logger.error(f"Avatar model {avatar_id} not found from {AGENT_MODEL_ROOT}")
            raise HTTPException(
                status_code=400, detail=f"Avatar model {avatar_id} not found"
            )

        connection_info = AvatarConnectionInfo(
            room_name=room_name,
            url=livekit_url,
            token=livekit_token,
            avatar_id=avatar_id,
            imx_path=imx_model_path,
        )
        worker = await launcher.launch_worker(connection_info)
        logger.info(f"Launched avatar worker for room: {connection_info.room_name}")

        # return {
        #     "status": "success",
        #     "message": f"Avatar worker launched for room: {connection_info.room_name}",
        # }
        
        # wait for the worker to exit for async mode
        tic = time.time()
        return_code = await worker.done_fut
        if return_code != 0:
            raise HTTPException(
                status_code=500, detail=f"Avatar worker exited with code {return_code}"
            )
        toc = time.time()
        logger.info(
            f"[{connection_info.room_name}] Avatar worker exited after {toc - tic:.2f} seconds"
        )
        return {
            "status": "success",
            "message": f"Avatar worker launched for room: {connection_info.room_name}",
            "duration": toc - tic,
        }
    except HTTPException:
        # Re-raise HTTP exceptions (from image processing failures)
        raise
    except Exception as e:
        logger.error(f"Error handling launch request: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to launch worker: {str(e)}"
        )  # noqa: B904
    finally:
        pass


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
