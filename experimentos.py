"""
Experimento comparativo — estratégias de gerenciamento de latência
====================================================================
Compara quatro condições sob a mesma carga de trabalho:

    (0) linha de base síncrona, sem estratégia
    (A) cache semântico
    (B) fila assíncrona com workers
    (C) streaming de resposta

Protocolo de medição:
  - função de temporização: time.perf_counter_ns() (monotônica, alta resolução)
  - Z execuções de aquecimento, descartadas
  - Y repetições medidas, reportadas como média +/- desvio padrão
  - semente fixa: a mesma carga é apresentada a todas as condições
"""

import platform
import statistics
import sys
import time

import numpy as np

import carga as mod_carga
import estrategias
import llm_stub

# =====================================================================
# PARÂMETROS DO EXPERIMENTO
# =====================================================================
N_REQUISICOES = 40           # requisições por execução
PROPORCAO_REPETICAO = 0.45   # fração da carga com equivalente semântico
N_REPETICOES = 30           # Y — repetições medidas
N_WARMUP = 5                 # Z — execuções de aquecimento (descartadas)
N_WORKERS = 8                # workers concorrentes da fila assíncrona
SEED = 42


def _media_dp(valores: list[float]) -> tuple[float, float]:
    if len(valores) < 2:
        return (float(valores[0]) if valores else 0.0), 0.0
    return statistics.mean(valores), statistics.stdev(valores)


def executar_condicao(nome: str, funcao, carga: list[dict]) -> dict:
    """Roda aquecimento + repetições de uma condição e consolida as métricas."""
    print(f"  {nome} ... ", end="", flush=True)

    for _ in range(N_WARMUP):
        funcao(carga)

    execucoes = [funcao(carga) for _ in range(N_REPETICOES)]

    consolidado = {"condicao": nome, "n_repeticoes": N_REPETICOES}
    chaves = execucoes[0].keys()
    for chave in chaves:
        valores = [e[chave] for e in execucoes]
        if isinstance(valores[0], (int, float)):
            media, dp = _media_dp([float(v) for v in valores])
            consolidado[chave] = media
            consolidado[chave + "_dp"] = dp

    print("ok")
    return consolidado


def exibir_resultados(resultados: list[dict]):
    print()
    print("=" * 96)
    print("RESULTADOS CONSOLIDADOS")
    print("=" * 96)
    cab = "{:<26} {:>22} {:>22} {:>16}"
    print(cab.format("Condição", "Latência total (ms)",
                     "1º retorno (ms)", "Throughput (req/s)"))
    print("-" * 96)
    for r in resultados:
        print(cab.format(
            r["condicao"],
            f"{r['latencia_media_ms']:.2f} ± {r['latencia_media_ms_dp']:.2f}",
            f"{r['latencia_primeiro_retorno_ms']:.2f} ± {r['latencia_primeiro_retorno_ms_dp']:.2f}",
            f"{r['throughput_req_s']:.2f}",
        ))
    print()

    # Métricas específicas do cache
    cache = next((r for r in resultados if "taxa_acerto_pct" in r), None)
    if cache:
        print("-" * 96)
        print("MÉTRICAS ESPECÍFICAS DO CACHE SEMÂNTICO")
        print("-" * 96)
        print(f"  Taxa de acerto (hit rate):        {cache['taxa_acerto_pct']:.2f}%")
        print(f"  Acertos verdadeiros:              {cache['acertos_verdadeiros']:.1f} req/execução")
        print(f"  Falsos positivos:                 {cache['falsos_positivos']:.1f} req/execução")
        print(f"  Precisão do cache:                {cache['precisao_cache_pct']:.2f}%")
        print(f"  Memória do índice vetorial:       {cache['memoria_estrutura_bytes']/1024:.1f} KB")
        print(f"  Limiar de similaridade adotado:   {estrategias.LIMIAR_SIMILARIDADE}")
        print()

    # Métricas específicas da fila
    fila = next((r for r in resultados if "espera_media_fila_ms" in r), None)
    if fila:
        print("-" * 96)
        print("MÉTRICAS ESPECÍFICAS DA FILA ASSÍNCRONA")
        print("-" * 96)
        print(f"  Workers concorrentes:             {int(fila['n_workers'])}")
        print(f"  Profundidade máxima da fila:      {int(fila['profundidade_maxima_fila'])} requisições")
        print(f"  Espera média em fila:             {fila['espera_media_fila_ms']:.2f} ± {fila['espera_media_fila_ms_dp']:.2f} ms")
        print()


def exibir_ambiente():
    print("=" * 96)
    print("AMBIENTE EXPERIMENTAL")
    print("=" * 96)
    print(f"  Sistema operacional:      {platform.system()} {platform.release()}")
    print(f"  Arquitetura:              {platform.machine()}")
    print(f"  Processador:              {platform.processor() or '[preencher: ver Gerenciador de Tarefas]'}")
    print(f"  Núcleos lógicos:          {__import__('os').cpu_count()}")
    print(f"  Python:                   {sys.version.split()[0]}")
    try:
        import psutil
        print(f"  Memória RAM total:        {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        print("  Memória RAM total:        [psutil não instalado — preencher manualmente]")

    import sklearn
    import scipy
    print(f"  Bibliotecas:              numpy {np.__version__}, scikit-learn {sklearn.__version__}, scipy {scipy.__version__}")
    print(f"  Função de temporização:   time.perf_counter_ns() (monotônica, alta resolução)")
    print(f"  Concorrência:             asyncio (event loop de thread única)")
    print()
    print("  Perfil de latência do stub de inferência:")
    print(f"    TTFT = {llm_stub.TTFT_MS} ms | {llm_stub.MS_POR_TOKEN} ms por token | "
          f"{llm_stub.TOKENS_MIN}-{llm_stub.TOKENS_MAX} tokens por resposta")
    print()


def gerar_grafico(resultados: list[dict], caminho="figura1_latencia_estrategias.png"):
    """Gera o gráfico de latência total vs. tempo até o primeiro retorno."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nomes = [r["condicao"] for r in resultados]
    total = [r["latencia_media_ms"] for r in resultados]
    total_dp = [r["latencia_media_ms_dp"] for r in resultados]
    primeiro = [r["latencia_primeiro_retorno_ms"] for r in resultados]
    primeiro_dp = [r["latencia_primeiro_retorno_ms_dp"] for r in resultados]

    throughput = [r["throughput_req_s"] for r in resultados]

    x = np.arange(len(nomes))
    largura = 0.38
    rotulos = [n.replace(" ", "\n") for n in nomes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    # (a) Latência: resposta completa vs. primeiro retorno
    ax1.bar(x - largura/2, total, largura, yerr=total_dp, capsize=2,
            label="Resposta completa", color="#D85A30")
    ax1.bar(x + largura/2, primeiro, largura, yerr=primeiro_dp, capsize=2,
            label="Primeiro retorno", color="#0F6E56")
    ax1.set_ylabel("Latência média (ms)", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rotulos, fontsize=7)
    ax1.legend(fontsize=7)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.4, linewidth=0.5)
    ax1.tick_params(labelsize=7)
    ax1.set_title("(a) Latência", fontsize=8)

    # (b) Throughput
    ax2.bar(x, throughput, largura * 1.6, color="#2E6FB7")
    ax2.set_ylabel("Throughput (req/s)", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(rotulos, fontsize=7)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4, linewidth=0.5)
    ax2.tick_params(labelsize=7)
    ax2.set_title("(b) Vazão", fontsize=8)

    plt.tight_layout(pad=0.6)
    plt.savefig(caminho, dpi=300)
    print(f"Gráfico salvo em: {caminho}")


CONDICOES = {
    "baseline": ("Linha de base", lambda c: estrategias.executar_baseline(c)),
    "cache": ("Cache semântico", lambda c: estrategias.executar_cache_semantico(c)),
    "fila": ("Fila assíncrona", lambda c: estrategias.executar_fila_assincrona(c, N_WORKERS)),
    "streaming": ("Streaming", lambda c: estrategias.executar_streaming(c)),
}

ARQUIVO_PARCIAIS = "resultados_parciais.json"


def _carregar_parciais() -> dict:
    import json
    import os
    if os.path.exists(ARQUIVO_PARCIAIS):
        with open(ARQUIVO_PARCIAIS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _salvar_parcial(chave: str, dados: dict):
    import json
    parciais = _carregar_parciais()
    parciais[chave] = dados
    with open(ARQUIVO_PARCIAIS, "w", encoding="utf-8") as f:
        json.dump(parciais, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Execução fracionada: python experimentos.py <condicao>
    # Sem argumento, executa todas as condições em sequência.
    if len(sys.argv) > 1 and sys.argv[1] in CONDICOES:
        chave = sys.argv[1]
        nome, funcao = CONDICOES[chave]
        carga_ = mod_carga.gerar_carga(N_REQUISICOES, PROPORCAO_REPETICAO, SEED)
        print(f"Executando condição isolada: {nome}")
        _salvar_parcial(chave, executar_condicao(nome, funcao, carga_))
        print(f"Resultado salvo em {ARQUIVO_PARCIAIS}")
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "consolidar":
        parciais = _carregar_parciais()
        ordem = ["baseline", "cache", "fila", "streaming"]
        resultados_ = [parciais[k] for k in ordem if k in parciais]
        carga_ = mod_carga.gerar_carga(N_REQUISICOES, PROPORCAO_REPETICAO, SEED)
        exibir_resultados(resultados_)
        exibir_ambiente()
        gerar_grafico(resultados_)
        sys.exit(0)

    print()
    print("=" * 96)
    print("EXPERIMENTO — GERENCIAMENTO DE LATÊNCIA EM APIs DE IA GENERATIVA")
    print("=" * 96)
    print(f"  Requisições por execução:     {N_REQUISICOES}")
    print(f"  Proporção com equivalente:    {PROPORCAO_REPETICAO*100:.0f}%")
    print(f"  Repetições medidas (Y):       {N_REPETICOES}")
    print(f"  Execuções de aquecimento (Z): {N_WARMUP}")
    print(f"  Workers da fila:              {N_WORKERS}")
    print(f"  Semente:                      {SEED}")
    print()

    carga = mod_carga.gerar_carga(N_REQUISICOES, PROPORCAO_REPETICAO, SEED)

    n_parafrase = sum(1 for i in carga if i["categoria"] == "parafrase")
    n_unica = sum(1 for i in carga if i["categoria"] == "unica")
    n_armadilha = sum(1 for i in carga if i["categoria"] == "armadilha")
    print(f"Composição da carga: {n_parafrase} paráfrases | {n_unica} únicas | {n_armadilha} armadilhas")
    print()

    # --- Etapa preliminar: calibração do limiar de similaridade ---
    print("=" * 96)
    print("CALIBRAÇÃO DO LIMIAR DE SIMILARIDADE DO CACHE")
    print("=" * 96)
    print("  Critério: adotar o menor limiar que preserva precisão de 100%,")
    print("  priorizando a correção da resposta sobre a taxa de reaproveitamento.")
    print()
    print("  {:>7}  {:>12}  {:>11}  {:>6}  {:>6}".format(
        "Limiar", "Cobertura", "Precisão", "VP", "FP"))
    print("  " + "-" * 50)
    for r in estrategias.calibrar_limiar(
            carga, [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]):
        print("  {:>7.2f}  {:>11.1f}%  {:>10.1f}%  {:>6d}  {:>6d}".format(
            r["limiar"], r["cobertura_pct"], r["precisao_pct"],
            r["verdadeiros"], r["falsos_positivos"]))
    print()
    print(f"  Limiar adotado no experimento: {estrategias.LIMIAR_SIMILARIDADE}")
    print()

    print("Executando condições (aquecimento + repetições):")
    resultados = [
        executar_condicao("Linha de base", estrategias.executar_baseline, carga),
        executar_condicao("Cache semântico", estrategias.executar_cache_semantico, carga),
        executar_condicao("Fila assíncrona",
                          lambda c: estrategias.executar_fila_assincrona(c, N_WORKERS), carga),
        executar_condicao("Streaming", estrategias.executar_streaming, carga),
    ]

    exibir_resultados(resultados)
    exibir_ambiente()
    gerar_grafico(resultados)
