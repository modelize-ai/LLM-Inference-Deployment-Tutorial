import json
from logging import getLogger, DEBUG, INFO
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from server.static_batching_server import get_server, Server, ServerConfig
from protocol.completion_task import (
    HuggingFaceCompletionInputs,
    HuggingFaceCompletionOutputs
)
from protocol.error import Error
from protocol.routes import (
    ROUTE_GET_MODEL_ID,
    ROUTE_POST_CONTINUOUS_BATCHING_COMPLETION,
    ROUTE_CLIENT_ONLY_POST_SERVER_STARTUP_EVENT,
    ROUTE_CLIENT_ONLY_POST_SERVER_SHUTDOWN_EVENT
)
from utils.log_util import RequestLoggingMiddleware


logger = getLogger("gunicorn.logger")  # by default, we use gunicorn to wrap the app


class AppConfig(BaseModel):
    model_id: str = Field(default=...)
    server_config_file_path: str = Field(default=...)
    client_url: Optional[str] = Field(default=None)
    debug: bool = Field(default=False)


APP_NAME = "LLM-Inference-SB-Server"

app = FastAPI(title=APP_NAME, version="0.1.0")
app_config: Optional[AppConfig] = None


def build_app(
    model_id: str = None,
    server_config_file_path: str = "sb_server_config.json",
    client_url: Optional[str] = None,
    debug: bool = False
):
    global app, app_config

    if model_id is None:
        raise ValueError("You must specify a real value to model_id.")

    logger.setLevel(DEBUG if debug else INFO)

    app_config = AppConfig(
        model_id=model_id,
        server_config_file_path=server_config_file_path,
        client_url=client_url,
        debug=debug
    )

    app.add_middleware(RequestLoggingMiddleware, logger=logger)

    return app


@app.on_event("startup")
def startup():
    # initialize server
    Server(
        config=ServerConfig(**json.load(open(app_config.server_config_file_path, "r", encoding="utf-8"))),
        logger=logger
    )
    # TODO: implement logic to inform client that server is startup


@app.on_event("shutdown")
def shutdown():
    pass  # TODO: implement logic to inform client that server is shutdown


@app.get(ROUTE_GET_MODEL_ID)
async def get_model_id():
    return JSONResponse(content=app_config.model_id)


@app.post(ROUTE_POST_CONTINUOUS_BATCHING_COMPLETION, response_model=HuggingFaceCompletionOutputs)
async def execute_completion(request_inputs: HuggingFaceCompletionInputs):
    server = get_server()
    return await server.wait_task_done(request_inputs)


if __name__ == "__main__":
    import uvicorn
    from argparse import ArgumentParser
    from logging import basicConfig

    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--server_config_file_path", type=str, default="sb_server_config.json")
    parser.add_argument("--client_url", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    logger = getLogger(__name__)  # override gunicorn logger if we use uvicorn directly
    basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    uvicorn.run(
        build_app(
            model_id=args.model_id,
            server_config_file_path=args.server_config_file_path,
            client_url=args.client_url,
            debug=args.debug
        ),
        host="0.0.0.0",
        port=args.port
    )
