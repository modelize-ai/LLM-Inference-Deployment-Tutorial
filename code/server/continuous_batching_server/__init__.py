from .config import *
from .server import *

DEFAULT_BLOCK_SIZE_A100 = 16
DEFAULT_NUM_BLOCKS_A100 = 2500
DEFAULT_BATCH_MAX_TOKENS_A100 = 56000
DEFAULT_BATCH_MAX_BEAMS_A100 = 32


__all__ = [
    "get_server",
    "Server",
    "BatcherConfig",
    "CacheConfig",
    "ModelLoadingConfig",
    "ParallelConfig",
    "ServerConfig",
    "DEFAULT_BLOCK_SIZE_A100",
    "DEFAULT_NUM_BLOCKS_A100",
    "DEFAULT_BATCH_MAX_TOKENS_A100",
    "DEFAULT_BATCH_MAX_BEAMS_A100"
]
