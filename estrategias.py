"""
Implementação das estratégias de gerenciamento de latência
============================================================
As três estratégias são efetivamente implementadas e medidas — apenas a
inferência do modelo é emulada pelo stub (ver llm_stub.py).

  A) CACHE SEMÂNTICO  — vetorização das perguntas e reaproveitamento de
     respostas por similaridade de cosseno acima de um limiar.
  B) FILA ASSÍNCRONA  — desacoplamento entre recebimento e processamento,
     com workers concorrentes consumindo uma fila.
  C) STREAMING        — emissão progressiva dos tokens ao cliente.

Todas são comparadas contra uma linha de base síncrona e sem estratégia.
"""

import asyncio
import time

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse

import llm_stub

# Limiar de similaridade para considerar duas perguntas equivalentes.
# Valor definido por calibracao empirica (ver calibrar_limiar): adota-se o
# menor limiar que preserva precisao de 100% na carga de calibracao, criterio
# apropriado a contextos institucionais, nos quais uma resposta incorreta
# devolvida pelo cache e mais danosa a confianca do usuario do que um miss.
LIMIAR_SIMILARIDADE = 0.90

# Dimensionalidade do espaço vetorial das perguntas.
N_FEATURES = 2 ** 14


def _ms(inicio_ns: int, fim_ns: int) -> float:
    return (fim_ns - inicio_ns) / 1_000_000.0


# =====================================================================
# LINHA DE BASE — síncrona, sem estratégia
# =====================================================================
def executar_baseline(carga: list[dict]) -> dict:
    """
    Processa as requisições sequencialmente, de forma bloqueante.
    O cliente não recebe nada até a resposta estar completa, portanto
    o tempo até o primeiro retorno é igual ao tempo total.
    """
    latencias = []
    inicio_total = time.perf_counter_ns()

    for item in carga:
        t0 = time.perf_counter_ns()
        llm_stub.gerar_bloqueante(item["texto"])
        latencias.append(_ms(t0, time.perf_counter_ns()))

    tempo_total_ms = _ms(inicio_total, time.perf_counter_ns())

    return {
        "latencia_media_ms": float(np.mean(latencias)),
        "latencia_primeiro_retorno_ms": float(np.mean(latencias)),  # = total
        "tempo_total_ms": tempo_total_ms,
        "throughput_req_s": len(carga) / (tempo_total_ms / 1000.0),
        "memoria_estrutura_bytes": 0,
    }


# =====================================================================
# A) CACHE SEMÂNTICO
# =====================================================================
class CacheSemantico:
    """
    Cache que reaproveita respostas de perguntas semanticamente próximas.

    A pergunta é projetada em um espaço vetorial por HashingVectorizer
    (sem etapa de treinamento, portanto sem vazamento de informação da
    carga futura) com normalização L2. Nessa condição, o produto interno
    entre dois vetores equivale à similaridade de cosseno.
    """

    def __init__(self, limiar: float = LIMIAR_SIMILARIDADE):
        self._vetorizador = HashingVectorizer(
            n_features=N_FEATURES,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            strip_accents="unicode",
        )
        self._limiar = limiar
        self._matriz = None            # vetores das perguntas armazenadas
        self._respostas: list[str] = []
        self._ids_semanticos: list[str] = []

    def _vetorizar(self, texto: str):
        return self._vetorizador.transform([texto])

    def consultar(self, texto: str):
        """Retorna (resposta, id_semantico_armazenado, similaridade) ou None."""
        if self._matriz is None:
            return None
        v = self._vetorizar(texto)
        similaridades = (self._matriz @ v.T).toarray().ravel()
        idx = int(np.argmax(similaridades))
        if similaridades[idx] >= self._limiar:
            return self._respostas[idx], self._ids_semanticos[idx], float(similaridades[idx])
        return None

    def armazenar(self, texto: str, resposta: str, id_semantico: str):
        v = self._vetorizar(texto)
        self._matriz = v if self._matriz is None else sparse.vstack([self._matriz, v])
        self._respostas.append(resposta)
        self._ids_semanticos.append(id_semantico)

    def memoria_bytes(self) -> int:
        """Memória ocupada pelo índice vetorial do cache."""
        if self._matriz is None:
            return 0
        m = self._matriz.tocsr()
        return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)


def executar_cache_semantico(carga: list[dict]) -> dict:
    """
    Processa a carga com cache semântico à frente do modelo.

    Além do desempenho, são contabilizados acertos verdadeiros e falsos
    positivos — casos em que o cache devolve a resposta de uma pergunta
    de conteúdo distinto, resultando em resposta incorreta ao usuário.
    """
    cache = CacheSemantico()
    latencias = []
    acertos_verdadeiros = 0
    falsos_positivos = 0
    misses = 0

    inicio_total = time.perf_counter_ns()

    for item in carga:
        t0 = time.perf_counter_ns()
        resultado = cache.consultar(item["texto"])

        if resultado is not None:
            _, id_armazenado, _ = resultado
            if id_armazenado == item["id_semantico"]:
                acertos_verdadeiros += 1
            else:
                falsos_positivos += 1
        else:
            misses += 1
            resposta = llm_stub.gerar_bloqueante(item["texto"])
            cache.armazenar(item["texto"], resposta, item["id_semantico"])

        latencias.append(_ms(t0, time.perf_counter_ns()))

    tempo_total_ms = _ms(inicio_total, time.perf_counter_ns())
    n = len(carga)
    total_hits = acertos_verdadeiros + falsos_positivos

    return {
        "latencia_media_ms": float(np.mean(latencias)),
        "latencia_primeiro_retorno_ms": float(np.mean(latencias)),
        "tempo_total_ms": tempo_total_ms,
        "throughput_req_s": n / (tempo_total_ms / 1000.0),
        "taxa_acerto_pct": 100.0 * total_hits / n,
        "acertos_verdadeiros": acertos_verdadeiros,
        "falsos_positivos": falsos_positivos,
        "precisao_cache_pct": (100.0 * acertos_verdadeiros / total_hits) if total_hits else 100.0,
        "misses": misses,
        "memoria_estrutura_bytes": cache.memoria_bytes(),
    }


def calibrar_limiar(carga: list[dict], limiares: list[float]) -> list[dict]:
    """
    Varre valores de limiar e mede, para cada um, a cobertura (fração de
    requisições atendidas pelo cache) e a precisão (fração dos atendimentos
    que devolveu a resposta correta).

    A classificação é avaliada sem chamar o modelo, pois apenas a decisão
    de acerto/erro é relevante aqui. Serve para escolher o limiar por
    critério explícito, em vez de arbitrá-lo.
    """
    resultados = []
    for limiar in limiares:
        cache = CacheSemantico(limiar=limiar)
        verdadeiros = falsos = 0
        for item in carga:
            r = cache.consultar(item["texto"])
            if r is not None:
                _, id_armazenado, _ = r
                if id_armazenado == item["id_semantico"]:
                    verdadeiros += 1
                else:
                    falsos += 1
            else:
                cache.armazenar(item["texto"], "[resposta]", item["id_semantico"])
        hits = verdadeiros + falsos
        resultados.append({
            "limiar": limiar,
            "cobertura_pct": 100.0 * hits / len(carga),
            "precisao_pct": (100.0 * verdadeiros / hits) if hits else 100.0,
            "verdadeiros": verdadeiros,
            "falsos_positivos": falsos,
        })
    return resultados


# =====================================================================
# B) FILA ASSÍNCRONA COM WORKERS
# =====================================================================
async def _executar_fila_async(carga: list[dict], n_workers: int) -> dict:
    """
    Todas as requisições são submetidas de uma vez (rajada), simulando um
    pico de acesso. Workers concorrentes consomem a fila. Mede-se a
    latência ponta a ponta, incluindo o tempo de espera em fila.
    """
    fila: asyncio.Queue = asyncio.Queue()
    latencias_e2e = []
    esperas_fila = []
    profundidade_maxima = 0

    inicio_total = time.perf_counter_ns()

    for item in carga:
        fila.put_nowait((item, time.perf_counter_ns()))
    profundidade_maxima = fila.qsize()

    async def worker():
        while True:
            try:
                item, t_enfileirado = fila.get_nowait()
            except asyncio.QueueEmpty:
                return
            t_inicio = time.perf_counter_ns()
            esperas_fila.append(_ms(t_enfileirado, t_inicio))
            await llm_stub.gerar_bloqueante_async(item["texto"])
            latencias_e2e.append(_ms(t_enfileirado, time.perf_counter_ns()))
            fila.task_done()

    await asyncio.gather(*[worker() for _ in range(n_workers)])

    tempo_total_ms = _ms(inicio_total, time.perf_counter_ns())

    return {
        "latencia_media_ms": float(np.mean(latencias_e2e)),
        "latencia_primeiro_retorno_ms": float(np.mean(latencias_e2e)),
        "espera_media_fila_ms": float(np.mean(esperas_fila)),
        "tempo_total_ms": tempo_total_ms,
        "throughput_req_s": len(carga) / (tempo_total_ms / 1000.0),
        "profundidade_maxima_fila": profundidade_maxima,
        "n_workers": n_workers,
        "memoria_estrutura_bytes": 0,
    }


def executar_fila_assincrona(carga: list[dict], n_workers: int) -> dict:
    return asyncio.run(_executar_fila_async(carga, n_workers))


# =====================================================================
# C) STREAMING DE RESPOSTA
# =====================================================================
def executar_streaming(carga: list[dict]) -> dict:
    """
    Processa a carga com emissão progressiva de tokens.

    Distingue-se aqui o tempo até o primeiro token — que é o que o usuário
    percebe como início da resposta — do tempo até a resposta completa.
    """
    ttft_list = []
    totais = []

    inicio_total = time.perf_counter_ns()

    for item in carga:
        t0 = time.perf_counter_ns()
        primeiro = True
        for _token in llm_stub.gerar_streaming(item["texto"]):
            if primeiro:
                ttft_list.append(_ms(t0, time.perf_counter_ns()))
                primeiro = False
        totais.append(_ms(t0, time.perf_counter_ns()))

    tempo_total_ms = _ms(inicio_total, time.perf_counter_ns())

    return {
        "latencia_media_ms": float(np.mean(totais)),
        "latencia_primeiro_retorno_ms": float(np.mean(ttft_list)),
        "tempo_total_ms": tempo_total_ms,
        "throughput_req_s": len(carga) / (tempo_total_ms / 1000.0),
        "memoria_estrutura_bytes": 0,
    }
