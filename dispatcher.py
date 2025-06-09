import json
import logging
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from livekit import api

load_dotenv()

logger = logging.getLogger("avatar-dispatcher")
logging.basicConfig(level=logging.INFO)

THIS_DIR = Path(__file__).parent.absolute()


@dataclass
class AvatarConnectionInfo:
    room_name: str
    url: str  # LiveKit server URL
    token: str  # Token for avatar worker to join


async def create_explicit_dispatch(conn_info: AvatarConnectionInfo):
    lkapi = api.LiveKitAPI()
    await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name="avatar_worker",
            room=conn_info.room_name,
            # TODO: add image url
            metadata=json.dumps({"url": conn_info.url, "token": conn_info.token}),
        )
    )
    logger.info("created dispatch for room %s", conn_info.room_name)
    await lkapi.aclose()


app = FastAPI(title="Avatar Dispatcher")


@app.post("/launch")
async def handle_launch(connection_info: AvatarConnectionInfo) -> dict:
    """Handle request to launch an avatar worker"""
    try:
        # TODO: accept uploading image
        await create_explicit_dispatch(connection_info)
        return {
            "status": "success",
            "message": f"Avatar worker launched for room: {connection_info.room_name}",
        }
    except Exception as e:
        import traceback

        logger.error(f"Error handling launch request: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Failed to launch worker: {str(e)}"
        )  # noqa: B904


def run_server(host: str = "0.0.0.0", port: int = 8089):
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", default=8089, help="Port to run server on")
    args = parser.parse_args()
    run_server(args.host, args.port)
