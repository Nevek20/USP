"""
Emulador de API de LLM com perfil de latência controlado
==========================================================
A inferência propriamente dita NÃO é executada: é emulada por um stub com
perfil de latência parametrizado em dois componentes, que é como APIs de
LLM efetivamente se comportam:

    latência_total = TTFT + (n_tokens x MS_POR_TOKEN)

onde TTFT (time to first token) é o tempo até o primeiro token e
MS_POR_TOKEN é o intervalo de emissão dos tokens seguintes.

Os valores absolutos são reduzidos por um fator de escala em relação aos
tempos típicos reportados para APIs de LLM (na ordem de segundos), de modo
a viabilizar a repetição do experimento. Como TODAS as estratégias
comparadas utilizam exatamente o mesmo stub, a comparação relativa entre
elas é preservada.

A chamada é modelada como operação limitada por E/S (I/O-bound), que é a
natureza real de uma requisição a uma API de LLM remota — daí o uso de
time.sleep / asyncio.sleep, que liberam a CPU durante a espera.
"""

import asyncio
import hashlib
import time

# --- Perfil de latência do stub (parâmetros explícitos e reprodutíveis) ---
TTFT_MS = 25.0          # tempo até o primeiro token
MS_POR_TOKEN = 3.0      # intervalo entre tokens subsequentes
TOKENS_MIN = 25         # nº mínimo de tokens da resposta
TOKENS_MAX = 45         # nº máximo de tokens da resposta


def _n_tokens(prompt: str) -> int:
    """
    Número de tokens da resposta, derivado deterministicamente do prompt.
    Garante que a mesma pergunta produza sempre a mesma resposta e o mesmo
    custo de tempo, tornando o experimento reprodutível.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    faixa = TOKENS_MAX - TOKENS_MIN + 1
    return TOKENS_MIN + (digest[0] % faixa)


def _resposta(prompt: str, n: int) -> str:
    return f"[resposta com {n} tokens para: {prompt[:40]}]"


# ---------------------------------------------------------------------
# Modo bloqueante (síncrono) — o cliente só recebe algo ao final
# ---------------------------------------------------------------------
def gerar_bloqueante(prompt: str) -> str:
    n = _n_tokens(prompt)
    total_s = (TTFT_MS + n * MS_POR_TOKEN) / 1000.0
    time.sleep(total_s)
    return _resposta(prompt, n)


# ---------------------------------------------------------------------
# Modo streaming — tokens são emitidos progressivamente
# ---------------------------------------------------------------------
def gerar_streaming(prompt: str):
    """Gerador que emite os tokens progressivamente (equivalente a SSE)."""
    n = _n_tokens(prompt)
    time.sleep(TTFT_MS / 1000.0)
    yield "token_0"
    for i in range(1, n):
        time.sleep(MS_POR_TOKEN / 1000.0)
        yield f"token_{i}"


# ---------------------------------------------------------------------
# Variantes assíncronas — usadas pela estratégia de filas
# ---------------------------------------------------------------------
async def gerar_bloqueante_async(prompt: str) -> str:
    n = _n_tokens(prompt)
    total_s = (TTFT_MS + n * MS_POR_TOKEN) / 1000.0
    await asyncio.sleep(total_s)
    return _resposta(prompt, n)


def latencia_teorica_ms(prompt: str) -> float:
    """Latência esperada do stub para um dado prompt (para verificação)."""
    return TTFT_MS + _n_tokens(prompt) * MS_POR_TOKEN
