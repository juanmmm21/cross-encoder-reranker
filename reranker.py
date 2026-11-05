"""Compatibilidad: el código vive en el paquete `cross_encoder_reranker`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `reranker`
directamente (otros repos de ai-core-infra lo hacen)."""

from cross_encoder_reranker.reranker import *  # noqa: F401,F403
from cross_encoder_reranker.reranker import CrossEncoderReranker  # noqa: F401
