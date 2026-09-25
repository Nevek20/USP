<div align="center">

# Gerenciamento de latência em APIs de IA generativa

Estudo comparativo entre **cache semântico**, **filas assíncronas** e **streaming de resposta**,<br>
com medição de latência, vazão e latência percebida.

Iniciação Científica apresentada no **SIICUSP 2026**

[![Resumo em PDF](https://img.shields.io/badge/Resumo-PDF-B01117?style=flat-square)](docs/USP.pdf)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3A5460?style=flat-square&logo=python&logoColor=white)

</div>

---

## A pergunta

A demora das APIs de modelos de linguagem é uma barreira real para uso interativo. Três estratégias de back-end costumam ser apontadas como solução, mas quase sempre comparadas de forma qualitativa.

Aqui elas foram implementadas e medidas sob a mesma carga para responder: **cada uma resolve o quê, e a que custo?**

## Resultado principal

As três atuam sobre dimensões diferentes do problema. Não competem entre si.

| Estratégia | Resposta completa | Primeiro retorno | Vazão |
|---|---:|---:|---:|
| Linha de base | 129,83 ms | 129,83 ms | 7,70 req/s |
| Cache semântico | **107,72 ms** | 107,72 ms | 9,28 req/s |
| Fila assíncrona | 390,49 ms | 390,49 ms | **54,72 req/s** |
| Streaming | 144,07 ms | **25,50 ms** | 6,94 req/s |

<sub>Média de 30 repetições, após 5 execuções de aquecimento.</sub>

- **Cache semântico** corta 17% da latência, mas só cobre 17,5% das requisições no limiar que garante 100% de precisão.
- **Fila assíncrona** multiplica a vazão por 7,1, ao custo de triplicar a latência individual sob rajada.
- **Streaming** não acelera o processamento (custa 11% a mais), mas entrega o primeiro trecho da resposta 80% mais cedo.

> O sistema não ficou mais rápido, mas o usuário vê a resposta começar quase de imediato. Essa é a diferença entre latência real e latência percebida.

## Um achado colateral

Ao calibrar o limiar de similaridade do cache, apareceu um problema que não estava previsto:

| Tipo de par | Similaridade |
|---|---:|
| Sentido oposto ("efetuar matrícula" x "cancelar matrícula") | **0,87** |
| Reformulações legítimas da mesma pergunta | 0,25 a 0,82 |

Na representação lexical usada, os pares mais perigosos parecem *mais* equivalentes que os pares corretos. Nenhum limiar separa os dois casos de forma limpa. Isso abriu uma linha de investigação sobre segurança de cache, hoje em desenvolvimento.

## Como reproduzir

Requisitos: Python 3.10 ou superior.

```bash
pip install scikit-learn scipy numpy matplotlib psutil
python experimentos.py
```

A execução completa leva alguns minutos.

<details>
<summary>Rodar por partes</summary>

<br>

```bash
python experimentos.py baseline
python experimentos.py cache
python experimentos.py fila
python experimentos.py streaming
python experimentos.py consolidar
```

Os resultados são acumulados em `resultados_parciais.json`, então as partes podem ser executadas em momentos diferentes.

</details>

## Organização do repositório

| Arquivo | Conteúdo |
|---|---|
| `llm_stub.py` | Emulador da API do modelo, com perfil de latência controlado |
| `carga.py` | Carga sintética: paráfrases, perguntas únicas e pares-armadilha |
| `estrategias.py` | As três estratégias, a linha de base e a calibração do limiar |
| `experimentos.py` | Execução das condições e geração da figura |
| `resultados_parciais.json` | Resultados da execução reportada no resumo |
| `docs/USP.pdf` | Resumo submetido ao SIICUSP |

## Metodologia

A inferência do modelo **não** é executada. Ela é emulada por um componente com perfil de latência parametrizado em tempo até o primeiro token e intervalo entre tokens. Os valores absolutos são reduzidos em escala frente a APIs reais, para viabilizar a repetição do experimento.

Já as **três estratégias são implementadas de verdade**: o cache vetoriza e compara as perguntas, a fila usa `asyncio` com workers concorrentes e o streaming emite os tokens progressivamente. Como todas as condições usam o mesmo emulador, a comparação entre elas se mantém válida.

- Tempos medidos com `time.perf_counter_ns()`, monotônica e de alta resolução
- Semente fixa: a mesma carga é apresentada a todas as condições
- Limiar do cache definido por calibração empírica, no menor valor que preserva 100% de precisão
- Nenhum dado pessoal real é utilizado

---

<div align="center">

**Autoria**<br>
João Guilherme Aguillera Oliveira · Matheus Guida

**Orientação**<br>
Prof. Dr. João Emmanuel D'Alkmin Neves

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/fatec-branca.png">
  <img src="docs/fatec.png" alt="Fatec Americana" width="180">
</picture>

<sub>Faculdade de Tecnologia de Americana Ministro Ralph Biasi</sub>

<br><br>

Dúvidas ou sugestões? Abra uma [issue](../../issues).

</div>
