"""
Geração da carga de trabalho sintética
========================================
A carga reproduz o padrão de uso de um assistente institucional, no qual
uma parcela das perguntas é semanticamente equivalente a perguntas já
feitas (reformulações da mesma dúvida), condição necessária para que o
cache semântico tenha efeito.

Três categorias de item compõem a carga:

  1. GRUPOS DE PARÁFRASE — perguntas distintas na forma, equivalentes no
     conteúdo. É o que o cache semântico deve reconhecer e reaproveitar.

  2. PERGUNTAS ÚNICAS — sem equivalente na carga. Devem sempre resultar em
     miss de cache.

  3. PARES ARMADILHA — perguntas lexicalmente próximas, porém de conteúdo
     distinto (ex.: efetuar x cancelar a mesma operação). Servem para medir
     falsos positivos do cache, isto é, respostas incorretas devolvidas por
     similaridade indevida.

Nenhum dado pessoal real é utilizado.
"""

import random

# --- 1. Grupos de paráfrase: mesmo id_semantico = mesma resposta esperada ---
GRUPOS_PARAFRASE = [
    ("rematricula", [
        "Como faço a rematrícula no próximo semestre?",
        "Como eu faço rematrícula para o semestre seguinte?",
        "Qual o procedimento para fazer a rematrícula do próximo semestre?",
        "Como realizar minha rematrícula no semestre que vem?",
    ]),
    ("historico", [
        "Onde emito o histórico escolar?",
        "Onde eu consigo emitir o meu histórico escolar?",
        "Como emitir o histórico escolar da faculdade?",
        "Onde faço a emissão do histórico escolar?",
    ]),
    ("trancamento", [
        "Qual o prazo para trancamento de matrícula?",
        "Até quando posso pedir o trancamento da matrícula?",
        "Qual é o prazo limite de trancamento de matrícula?",
    ]),
    ("bolsa", [
        "Como solicito bolsa de estudos na instituição?",
        "Qual o procedimento para solicitar bolsa de estudos?",
        "Como fazer a solicitação de bolsa de estudos?",
    ]),
    ("estagio", [
        "Como registrar meu contrato de estágio?",
        "Qual o procedimento para registrar o contrato de estágio?",
        "Onde faço o registro do meu contrato de estágio?",
    ]),
]

# --- 2. Perguntas únicas (sem equivalente na carga) ---
PERGUNTAS_UNICAS = [
    "Qual o horário de funcionamento da biblioteca aos sábados?",
    "Quantos créditos preciso para colar grau em engenharia?",
    "O restaurante universitário aceita pagamento por aproximação?",
    "Existe convênio de intercâmbio com universidades de Portugal?",
    "Como funciona a validação de disciplinas cursadas em outra faculdade?",
    "Qual o telefone da ouvidoria da instituição?",
    "Há vagas de monitoria abertas para o departamento de computação?",
    "Onde fica o setor de assistência estudantil no campus?",
    "Qual o valor da taxa de emissão da segunda via do diploma?",
    "É possível cursar disciplinas em outro turno?",
    "Como faço para acessar a rede wi-fi do campus?",
    "Quais documentos preciso levar no dia da matrícula presencial?",
    "O estacionamento do campus tem vagas para motocicletas?",
    "Quem é o coordenador do curso de análise de sistemas?",
    "A instituição oferece curso de extensão em libras?",
    "Qual a nota mínima de aprovação em cada disciplina?",
    "Posso levar acompanhante na cerimônia de formatura?",
    "Existe atendimento psicológico gratuito para estudantes?",
    "Como participo do programa de iniciação científica?",
    "A carteirinha estudantil dá desconto no transporte público?",
    "Onde consulto o calendário acadêmico deste ano?",
    "Quantas faltas são permitidas por disciplina no semestre?",
    "O laboratório de informática abre no período noturno?",
    "Há armários disponíveis para alugar no bloco central?",
    "Qual o procedimento em caso de perda do crachá de acesso?",
    "A faculdade tem grupo de pesquisa em energias renováveis?",
    "Como faço para trancar apenas uma disciplina isolada?",
    "Existe auxílio transporte para estudantes de baixa renda?",
]

# --- 3. Pares armadilha: alta similaridade lexical, conteúdo distinto ---
PARES_ARMADILHA = [
    ("efetuar_matricula", "Como faço a matrícula na disciplina optativa?"),
    ("cancelar_matricula", "Como faço o cancelamento da matrícula na disciplina optativa?"),
    ("prazo_inicio_bolsa", "Qual a data de início do pagamento da bolsa?"),
    ("prazo_fim_bolsa", "Qual a data de encerramento do pagamento da bolsa?"),
    ("abrir_recurso", "Qual o prazo para abrir recurso contra a nota da prova?"),
    ("responder_recurso", "Qual o prazo para a resposta do recurso contra a nota da prova?"),
    ("entrada_intercambio", "Quais requisitos para entrar no programa de intercâmbio?"),
    ("saida_intercambio", "Quais requisitos para sair do programa de intercâmbio?"),
]


def gerar_carga(n_requisicoes: int, proporcao_repeticao: float, seed: int) -> list[dict]:
    """
    Monta a sequência de requisições.

    proporcao_repeticao — fração da carga composta por perguntas que possuem
    equivalente semântico anterior (parâmetro da carga de trabalho, e não do
    mecanismo avaliado).

    Cada item traz o campo id_semantico, que é o gabarito usado para
    classificar acertos e falsos positivos do cache.
    """
    rng = random.Random(seed)
    itens: list[dict] = []

    n_repetidas = int(n_requisicoes * proporcao_repeticao)
    n_armadilha = min(len(PARES_ARMADILHA), max(2, int(n_requisicoes * 0.15)))
    n_unicas = n_requisicoes - n_repetidas - n_armadilha

    # Perguntas com equivalente semântico (paráfrases)
    for i in range(n_repetidas):
        id_sem, variantes = GRUPOS_PARAFRASE[i % len(GRUPOS_PARAFRASE)]
        itens.append({
            "texto": rng.choice(variantes),
            "id_semantico": id_sem,
            "categoria": "parafrase",
        })

    # Perguntas únicas — nunca são geradas variações artificiais: se a carga
    # exigir mais itens do que o repertório disponível, o excedente é
    # completado com paráfrases, preservando a integridade dos rótulos.
    n_unicas_reais = min(max(0, n_unicas), len(PERGUNTAS_UNICAS))
    for i in range(n_unicas_reais):
        itens.append({
            "texto": PERGUNTAS_UNICAS[i],
            "id_semantico": f"unica_{i}",
            "categoria": "unica",
        })

    excedente = max(0, n_unicas - n_unicas_reais)
    for i in range(excedente):
        id_sem, variantes = GRUPOS_PARAFRASE[i % len(GRUPOS_PARAFRASE)]
        itens.append({
            "texto": rng.choice(variantes),
            "id_semantico": id_sem,
            "categoria": "parafrase",
        })

    # Pares armadilha
    for id_sem, texto in PARES_ARMADILHA[:n_armadilha]:
        itens.append({
            "texto": texto,
            "id_semantico": id_sem,
            "categoria": "armadilha",
        })

    rng.shuffle(itens)
    return itens
