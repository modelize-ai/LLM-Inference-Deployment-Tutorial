import logging
import re
from datetime import datetime
from typing import *

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# adopt from:
# https://github.com/jeffsiver/fastapi-route-logger/blob/main/fastapi_route_logger_middleware/__init__.py
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: FastAPI,
        *,
        logger: Optional[logging.Logger] = None,
        skip_routes: List[str] = None,
        skip_regexes: List[str] = None,
        skip_request_methods: List[str] = None,
    ):
        self._logger = logger if logger else logging.getLogger(__name__)
        self._skip_routes = skip_routes if skip_routes else []
        self._skip_regexes = (
            list(map(lambda regex: re.compile(regex), skip_regexes))
            if skip_regexes
            else []
        )
        self._skip_request_methods = skip_request_methods if skip_request_methods else []

        BaseHTTPMiddleware.__init__(self, app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if self._should_route_be_skipped(request):
            return await call_next(request)

        return await self._execute_request_with_logging(request, call_next)

    def _should_route_be_skipped(self, request: Request) -> bool:
        return any(
            [True for path in self._skip_routes if request.url.path.startswith(path)]
            + [True for regex in self._skip_regexes if regex.match(request.url.path)]
            + [True for method in self._skip_request_methods if request.method.lower() == method.lower()]
        )

    async def _execute_request_with_logging(
        self, request: Request, call_next: Callable
    ) -> Response:
        self._logging_before_execution(request)
        start_time = datetime.utcnow()
        try:
            response = await call_next(request)
        except Exception as e:
            self._logging_when_error_raised(e)
            raise

        finish_time = datetime.utcnow()
        self._logging_after_execution(response, (finish_time - start_time).total_seconds())

        return response

    def _logging_before_execution(self, request: Request):
        content = (
            f"receive a request from {request.client.host}:{request.client.port} "
            f"to {request.url.path}, method={request.method}"
        )
        self._logger.info(content)

    def _logging_after_execution(self, response: Response, execution_time: float):
        overall_status = "successfully" if response.status_code < 400 else "failed"
        content = (
            f"{overall_status} executed a request, duration={execution_time}s, "
            f"status_code={response.status_code}"
        )
        self._logger.info(content)

    def _logging_when_error_raised(self, exception: Exception):
        content = (
            f"error occurred when execute request, "
            f"error_type=[{exception.__class__.__name__}], error_msg=[{str(exception)}]"
        )
        self._logger.error(content)


__all__ = ["RequestLoggingMiddleware"]
