import torch

BASE_SENTENCES = [

    # BAIXO RISCO (0)
    "paciente com dor leve",
    "paciente com desconforto leve",
    "quadro estavel sem dor",
    "sem sintomas relevantes",
    "condicao estavel sem risco",
    "paciente consciente e orientado",
    "leve tontura sem agravamento",
    "sem sinais de hemorragia",
    "paciente com febre baixa controlada",
    "estado geral bom",
    "sem dificuldade respiratoria",
    "paciente caminhando normalmente",
    "leve dor abdominal sem piora",
    "quadro sem agravamento recente",
    "sintomas leves e controlados",
    "paciente tranquilo e responsivo",
    "sem risco iminente",
    "dor moderada controlada por medicacao",
    "paciente hidratado e estavel",
    "sem sinais de infeccao grave",
    "estado clinico estavel",
    "quadro controlado sem complicacoes",
    "sem alteracoes neurologicas",
    "paciente em repouso sem queixas",
    "leve mal estar geral",
    "mal estar geral muito leve",
    "condicao clinica estavel",
    "sem risco de vida imediato",
    "paciente alerta e calmo",
    "sem sinais de choque",
    "paciente cooperativo",
    "dor leve relatada",


    # ALTO RISCO (1)
    "paciente com dor intensa",
    "quadro critico com risco elevado",
    "sintomas graves e dor intensa",
    "paciente ofegante com dificuldade respiratoria",
    "hemorragia ativa detectada",
    "paciente inconsciente",
    "quadro de choque iminente",
    "dor aguda intensa no peito",
    "paciente com parada respiratoria",
    "sinais de insuficiencia respiratoria",
    "queda de pressao acentuada",
    "paciente com convulsoes",
    "perda de consciencia recente",
    "insuficiencia cardiaca aguda",
    "quadro instavel com risco de morte",
    "estado critico geral",
    "dor intensa abdominal persistente",
    "sangramento severo",
    "paciente com dispneia grave",
    "colapso circulatorio",
    "estado grave com evolução rapida",
    "sinais claros de sepse",
    "comprometimento neurologico severo",
    "paciente em parada cardiorrespiratoria",
    "quadro de emergencia medica",
    "hipoxia severa detectada",
    "instabilidade hemodinamica",
    "paciente sem resposta a estimulos",
    "estado geral crítico",
    "estado geral muito grave",
    "dor insuportavel relatada",
    "estado muito grave",

]

# Agora expandindo sistematicamente (~200 total)

def generate_variations(base_list, label):
    variations = []
    for sentence in base_list:
        variations.append(sentence)

        # variações simples controladas
        variations.append(sentence + " com leve evolucao")
        variations.append(sentence + " sem agravamento recente")
        variations.append("historico de " + sentence)
        variations.append("avaliacao indica " + sentence)

    labels = [label] * len(variations)
    return variations, labels


low_base = BASE_SENTENCES[:32]
high_base = BASE_SENTENCES[32:]

low_sentences, low_labels = generate_variations(low_base, 0)
high_sentences, high_labels = generate_variations(high_base, 1)

SENTENCES = low_sentences + high_sentences
LABELS = torch.tensor(low_labels + high_labels)

print(len(SENTENCES))  # ~200+

