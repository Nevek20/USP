# Gerenciamento de latência em APIs de IA generativa

Estudo comparativo entre **cache semântico**, **filas assíncronas** e **streaming de resposta**, com medição de latência, vazão e latência percebida.

Trabalho de Iniciação Científica apresentado no **SIICUSP 2026**.

### [Leia o resumo completo clicando aqui!](docs/USP.pdf)

---

## O que este trabalho responde

A demora das APIs de modelos de linguagem é uma barreira real para uso interativo. Três estratégias de back-end são apontadas como solução, mas costumam ser comparadas de forma qualitativa.

Aqui elas foram implementadas e medidas sob a mesma carga, para responder: **cada uma resolve o quê, e a que custo?**

## Resultado principal

As três atuam sobre dimensões diferentes do problema. Não competem entre si.

| Estratégia | Resposta completa | Primeiro retorno | Vazão |
|---|---|---|---|
| Linha de base | 129,83 ms | 129,83 ms | 7,70 req/s |
| Cache semântico | **107,72 ms** | 107,72 ms | 9,28 req/s |
| Fila assíncrona | 390,49 ms | 390,49 ms | **54,72 req/s** |
| Streaming | 144,07 ms | **25,50 ms** | 6,94 req/s |

*Média de 30 repetições, após 5 execuções de aquecimento.*

Em resumo:

- **Cache semântico** corta 17% da latência, mas só cobre 17,5% das requisições no limiar que garante 100% de precisão.
- **Fila assíncrona** multiplica a vazão por 7,1, ao custo de triplicar a latência individual sob rajada.
- **Streaming** não acelera o processamento (na verdade custa 11% a mais), mas entrega o primeiro trecho da resposta 80% mais cedo.

O último ponto delimita a diferença entre latência real e latência percebida: o sistema não ficou mais rápido, mas o usuário vê a resposta começar quase de imediato.

## Um achado colateral

Ao calibrar o limiar de similaridade do cache, apareceu um problema que não estava previsto.

Perguntas de **sentido oposto** (por exemplo, "efetuar matrícula" e "cancelar matrícula") atingiram similaridade de 0,87, enquanto reformulações legítimas da mesma pergunta ficaram entre 0,25 e 0,82.

Ou seja: na representação lexical usada, os pares mais perigosos parecem *mais* equivalentes que os pares corretos. Não existe limiar que separe os dois casos de forma limpa. Isso levantou uma linha de investigação sobre segurança de cache, hoje em desenvolvimento.

## Como reproduzir

Requisitos: Python 3.10 ou superior.

```bash
pip install scikit-learn scipy numpy matplotlib psutil
python experimentos.py
```

A execução completa leva alguns minutos. Para acompanhar o progresso, rode por partes:

```bash
python experimentos.py baseline
python experimentos.py cache
python experimentos.py fila
python experimentos.py streaming
python experimentos.py consolidar
```

Os resultados são acumulados em `resultados_parciais.json`, então as partes podem ser executadas em momentos diferentes.

## Organização do repositório

| Arquivo | Conteúdo |
|---|---|
| `llm_stub.py` | Emulador da API do modelo, com perfil de latência controlado |
| `carga.py` | Carga sintética: paráfrases, perguntas únicas e pares-armadilha |
| `estrategias.py` | As três estratégias, a linha de base e a calibração do limiar |
| `experimentos.py` | Execução das condições e geração da figura |
| `resultados_parciais.json` | Resultados da execução reportada no resumo |
| `docs/resumo-siicusp.pdf` | Resumo submetido ao SIICUSP |

## Sobre a metodologia

A inferência do modelo **não** é executada: é emulada por um componente com perfil de latência parametrizado em tempo até o primeiro token e intervalo entre tokens subsequentes. Os valores absolutos são reduzidos em escala frente aos tempos típicos de APIs reais, para viabilizar a repetição do experimento.

As **três estratégias são implementadas de verdade**: o cache vetoriza e compara as perguntas, a fila usa `asyncio` com workers concorrentes, o streaming emite os tokens progressivamente. Como todas as condições usam o mesmo emulador, a comparação entre elas se mantém válida.

Detalhes do protocolo:

- Tempos medidos com `time.perf_counter_ns()`, função monotônica de alta resolução
- Semente fixa: a mesma carga é apresentada a todas as condições
- O limiar do cache foi definido por calibração empírica, adotando o menor valor que preserva precisão de 100%
- Nenhum dado pessoal real é utilizado

## Autoria

João Guilherme Aguillera Oliveira

Matheus Guida

Orientação: Prof. Dr. João Emmanuel D'Alkmin Neves

Faculdade de Tecnologia de Americana Ministro Ralph Biasi

## Contato

Dúvidas ou sugestões: abra uma [issue](../../issues).
