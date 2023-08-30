import json
import time
import os
from logging import getLogger, DEBUG, INFO, Logger
from typing import Optional
from threading import Thread


from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from client import get_client, Client, ClientConfig, ServerType
from protocol.completion_task import *
from protocol.error import Error
from protocol.routes import (
    ROUTE_POST_OPENAI_CHAT_COMPLETION,
    ROUTE_POST_CONTINUOUS_BATCHING_COMPLETION,
    ROUTE_POST_STATIC_BATCHING_COMPLETION,
    ROUTE_CLIENT_ONLY_POST_SERVER_STARTUP_EVENT,
    ROUTE_CLIENT_ONLY_POST_SERVER_SHUTDOWN_EVENT
)
from utils.log_util import RequestLoggingMiddleware


logger = getLogger("gunicorn.logger")  # by default, we use gunicorn to wrap the app


class AppConfig(BaseModel):
    client_config_file_path: str = Field(default=...)
    client_config_hot_update_interval_minutes: int = Field(default=1)
    debug: bool = Field(default=False)


APP_NAME = "LLM-Inference-Client"

app = FastAPI(title=APP_NAME, version="0.1.0")
app_config: Optional[AppConfig] = None


def build_app(
    client_config_file_path: str = "client_config.json",
    client_config_hot_update_interval_minutes: int = 1,
    debug: bool = False
):
    global app, app_config

    logger.setLevel(DEBUG if debug else INFO)

    app_config = AppConfig(
        client_config_file_path=client_config_file_path,
        client_config_hot_update_interval_minutes=client_config_hot_update_interval_minutes,
        debug=debug
    )

    app.add_middleware(RequestLoggingMiddleware, logger=logger)

    return app


def hot_update_client_config_loop():
    while True:
        time.sleep(app_config.client_config_hot_update_interval_minutes * 60)
        client = get_client()

        fp = app_config.client_config_file_path
        if not os.path.exists(fp):
            logger.warning(
                msg=f"Client config file path [{fp}] not exists, skip hot update this time,"
                    f"and will try to save a client config snapshot."
            )
            try:
                client.save_config(fp)
            except:
                pass
            continue
        new_client_config = Client(**json.load(open(fp, "r", encoding="utf-8")))
        client.update_config(new_client_config)


@app.on_event("startup")
def startup():
    # init client
    Client(
        config=ClientConfig(
            **json.load(open(app_config.client_config_file_path, "r", encoding="utf-8"))
        ),
        logger=logger
    )

    # start client config hot update loop
    Thread(target=hot_update_client_config_loop, daemon=True).start()


# === Routes that request to LLM servers === #

@app.post(ROUTE_POST_OPENAI_CHAT_COMPLETION, response_model=OpenAIChatCompletionOutputs)
async def request_openai_chat_completion(request_inputs: OpenAIChatCompletionInputs):
    client = get_client()
    return await client.openai_chat_completion(request_inputs)


@app.post(ROUTE_POST_CONTINUOUS_BATCHING_COMPLETION, response_model=HuggingFaceCompletionOutputs)
async def request_continuous_batching_server(request_inputs: HuggingFaceCompletionInputs):
    client = get_client()
    return await client.huggingface_completion(request_inputs, server_type=ServerType.CB)


@app.post(ROUTE_POST_STATIC_BATCHING_COMPLETION, response_model=HuggingFaceCompletionOutputs)
async def request_static_batching_server(request_inputs: HuggingFaceCompletionInputs):
    client = get_client()
    return await client.huggingface_completion(request_inputs, server_type=ServerType.SB)


# === Routes that provide some meta information === #

@app.get("/cb_server/available_models")
async def get_cb_server_available_models():
    client = get_client()
    available_models = []
    for model_id, server_urls in client.model_id2continuous_batching_server_urls:
        if any(url_obj.available for url_obj in server_urls):
            available_models.append(model_id)
    return JSONResponse(content=available_models)


@app.get("/sb_server/available_models")
async def get_sb_server_available_models():
    client = get_client()
    available_models = []
    for model_id, server_urls in client.model_id2static_batching_server_urls:
        if any(url_obj.available for url_obj in server_urls):
            available_models.append(model_id)
    return JSONResponse(content=available_models)


@app.get("/openai_jumper/is_available")
async def get_openai_jumper_is_available():
    client = get_client()
    if any(jumper.available for jumper in client.openai_jumpers):
        return JSONResponse(content="1")
    return JSONResponse("0")


# === Routes that receive server event and update client config === #
# TODO: Implement routes


if __name__ == "__main__":
    import uvicorn
    from argparse import ArgumentParser
    from logging import basicConfig

    parser = ArgumentParser()
    parser.add_argument("--client_config_file_path", type=str, default="client_config.json")
    parser.add_argument("--client_config_hot_update_interval_minutes", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    logger = getLogger(__name__)  # override gunicorn logger if we use uvicorn directly
    basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    uvicorn.run(
        build_app(
            client_config_file_path=args.client_config_file_path,
            client_config_hot_update_interval_minutes=args.client_config_hot_update_interval_minutes,
            debug=args.debug,
        ),
        host="0.0.0.0",
        port=args.port
    )
