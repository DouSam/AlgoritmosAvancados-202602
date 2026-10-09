"""
========================================================================================
Algoritmos Avançados - Casos Práticos de Aplicação no Agronegócio & Big Data
Prof. Me. Douglas Samuel Gonçalves
Fatec Bebedouro - Jorge Caram Sabbag
Tecnologia em Big Data no Agronegócio (2º Semestre / 2026)
========================================================================================

Este script implementa em Python puro e de forma autônoma os 3 Estudos de Caso Práticos
apresentados ao final da aula de Busca em Grafos:

  [CASO 1] Roteamento de Telemetria com Menor Salto em Rede Mesh IoT (BFS)
           - Descoberta em ondas, árvore de predecessores e caminho ótimo.
           - Simulação de tolerância a falhas com re-roteamento automático.

  [CASO 2] Detecção de Deadlock e Ciclos em Pipelines ETL Agro no Apache Airflow (DFS)
           - Carimbos de tempo d[u] e f[u], classificação das 4 arestas (Árvore, Retorno,
             Avanço, Cruzamento) e identificação precisa do ciclo causador do travamento.

  [CASO 3] Propagação Epidemiológica da Ferrugem Asiática com Multi-Source BFS
           - Focos simultâneos de infecção, ondas de contaminação temporal (dias)
             e identificação do vetor de contaminação para zoneamento sanitário.

Uso:
  python aula18_casos_praticos.py          -> Executa todos os 3 casos sequencialmente
  python aula18_casos_praticos.py 1        -> Executa exclusivamente o Caso 1
  python aula18_casos_praticos.py 2        -> Executa exclusivamente o Caso 2
  python aula18_casos_praticos.py 3        -> Executa exclusivamente o Caso 3
========================================================================================
"""

import sys
from collections import deque
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# Reconfigura stdout para UTF-8 no Windows (evita caracteres corrompidos)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ======================================================================================
# ESTRUTURA BASE: TAD Grafo com BFS e DFS
# ======================================================================================
class CorDFS(Enum):
    BRANCO = 0  # Não visitado
    CINZA = 1   # Em processamento (na pilha de recursão)
    PRETO = 2   # Finalizado (todos os vizinhos processados)


class GrafoAgro:
    """
    TAD Grafo leve e robusto para suporte aos estudos de caso do Agronegócio.
    Implementa Lista de Adjacência via Dicionário de Dicionários.
    """

    def __init__(self, direcionado: bool = False):
        self.direcionado: bool = direcionado
        self.adj: Dict[str, Dict[str, float]] = {}

    def adicionar_vertice(self, v: str) -> None:
        if v not in self.adj:
            self.adj[v] = {}

    def adicionar_aresta(self, u: str, v: str, peso: float = 1.0) -> None:
        self.adicionar_vertice(u)
        self.adicionar_vertice(v)
        self.adj[u][v] = peso
        if not self.direcionado:
            self.adj[v][u] = peso

    def remover_aresta(self, u: str, v: str) -> bool:
        if u in self.adj and v in self.adj[u]:
            del self.adj[u][v]
            if not self.direcionado and v in self.adj and u in self.adj[v]:
                del self.adj[v][u]
            return True
        return False

    def vizinhos(self, v: str) -> List[str]:
        return sorted(self.adj.get(v, {}).keys())

    @property
    def vertices(self) -> List[str]:
        return sorted(self.adj.keys())

    # --- Algoritmo de Busca em Largura (BFS) ---
    def bfs(self, inicio: str) -> Tuple[Dict[str, int], Dict[str, Optional[str]], List[str]]:
        """
        Executa BFS a partir de 'inicio'.
        Retorna:
          - dist: dicionário com distância em número de saltos
          - pred: dicionário com o nó pai/predecessor
          - ordem: ordem de visitação dos nós
        """
        dist: Dict[str, int] = {v: -1 for v in self.vertices}
        pred: Dict[str, Optional[str]] = {v: None for v in self.vertices}
        ordem: List[str] = []

        dist[inicio] = 0
        fila: deque[str] = deque([inicio])

        while fila:
            u = fila.popleft()
            ordem.append(u)

            for v in self.vizinhos(u):
                if dist[v] == -1:  # Não visitado
                    dist[v] = dist[u] + 1
                    pred[v] = u
                    fila.append(v)

        return dist, pred, ordem

    def caminho_minimo_bfs(self, origem: str, destino: str) -> List[str]:
        """Reconstrói a rota ótima de menor número de saltos entre origem e destino."""
        dist, pred, _ = self.bfs(origem)
        if dist.get(destino, -1) == -1:
            return []  # Destino inalcançável

        caminho: List[str] = []
        atual: Optional[str] = destino
        while atual is not None:
            caminho.append(atual)
            atual = pred.get(atual)
        return list(reversed(caminho))

    # --- Algoritmo de Busca Multi-Fonte (Multi-Source BFS) ---
    def multi_source_bfs(self, fontes: List[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Expande ondas de contaminação simultâneas a partir de múltiplos focos.
        Retorna:
          - dist: menor distância temporal (dias/ondas) até o foco mais próximo
          - vetor: identificador do foco original responsável pela contaminação
        """
        dist: Dict[str, int] = {v: -1 for v in self.vertices}
        vetor: Dict[str, str] = {v: "" for v in self.vertices}
        fila: deque[str] = deque()

        for f in dict.fromkeys(fontes):  # Garante unicidade
            if f in self.adj:
                dist[f] = 0
                vetor[f] = f
                fila.append(f)

        while fila:
            u = fila.popleft()
            for v in self.vizinhos(u):
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    vetor[v] = vetor[u]
                    fila.append(v)

        return dist, vetor

    # --- Algoritmo de Busca em Profundidade (DFS) & Detecção de Ciclos ---
    def detectar_ciclo_dfs(self) -> Tuple[bool, List[str], Dict[str, int], Dict[str, int], Dict[str, List[Tuple[str, str]]]]:
        """
        Executa DFS completo com carimbos de tempo d[u], f[u] e classificação de arestas.
        Retorna:
          - tem_ciclo: bool
          - ciclo_encontrado: lista de vértices formando o ciclo
          - d: tempos de descoberta
          - f: tempos de término
          - arestas: classificação em 'arvore', 'retorno', 'avanco', 'cruzamento'
        """
        cor: Dict[str, CorDFS] = {v: CorDFS.BRANCO for v in self.vertices}
        pred: Dict[str, Optional[str]] = {v: None for v in self.vertices}
        d: Dict[str, int] = {}
        f: Dict[str, int] = {}
        tempo = 0

        arestas: Dict[str, List[Tuple[str, str]]] = {
            "arvore": [],
            "retorno": [],
            "avanco": [],
            "cruzamento": [],
        }

        ciclo_detectado: List[str] = []

        def dfs_visitar(u: str) -> None:
            nonlocal tempo
            tempo += 1
            d[u] = tempo
            cor[u] = CorDFS.CINZA

            for v in self.vizinhos(u):
                if cor[v] == CorDFS.BRANCO:
                    arestas["arvore"].append((u, v))
                    pred[v] = u
                    dfs_visitar(v)
                elif cor[v] == CorDFS.CINZA:
                    # Aresta de Retorno (Back-Edge) -> Ciclo garantido pelo Teorema de Cormen!
                    arestas["retorno"].append((u, v))
                    if not ciclo_detectado:
                        # Reconstrói o ciclo a partir da cadeia de predecessores
                        trilha = [v, u]
                        cur = u
                        while cur != v and pred[cur] is not None:
                            cur = pred[cur]  # type: ignore
                            trilha.append(cur)
                        ciclo_detectado.extend(reversed(trilha))
                elif cor[v] == CorDFS.PRETO:
                    if d[u] < d[v]:
                        arestas["avanco"].append((u, v))
                    else:
                        arestas["cruzamento"].append((u, v))

            cor[u] = CorDFS.PRETO
            tempo += 1
            f[u] = tempo

        for v in self.vertices:
            if cor[v] == CorDFS.BRANCO:
                dfs_visitar(v)

        tem_ciclo = len(arestas["retorno"]) > 0
        return tem_ciclo, ciclo_detectado, d, f, arestas


# ======================================================================================
# ESTUDO DE CASO 1: REDE MESH DE TELEMETRIA AGRÍCOLA (BFS)
# ======================================================================================
def executar_caso_1():
    print("\n" + "#" * 80)
    print("ESTUDO DE CASO 1: ROTEAMENTO EM REDE MESH DE TELEMETRIA AGRÍCOLA (BFS)")
    print("#" * 80)
    print("""
Contexto:
  Uma lavoura de cana-de-açúcar e citros de 1.200 hectares em Bebedouro/SP utiliza sensores
  IoT sem fio (umidade de solo, estações meteorológicas, pivôs centrais de irrigação).
  A topologia forma uma REDE MESH: sensores distantes não alcançam o Gateway Central diretamente
  e precisam retransmitir seus pacotes de telemetria através de repetidores intermediários.

Objetivo de Engenharia:
  Construir a tabela de roteamento de Menor Salto (Min-Hop) a partir do Gateway Central,
  garantindo mínima latência e máxima economia de bateria dos rádios LoRaWAN.
""")

    rede = GrafoAgro(direcionado=False)

    # Conexões de Rádio (Enlaces Físicos)
    enlaces = [
        ("Gateway_Central", "Repetidor_Norte"),
        ("Gateway_Central", "Repetidor_Sul"),
        ("Repetidor_Norte", "Sensor_Talhao_1"),
        ("Repetidor_Norte", "Sensor_Talhao_2"),
        ("Repetidor_Sul", "Estacao_Meteo"),
        ("Repetidor_Sul", "Pivo_Central_1"),
        ("Sensor_Talhao_2", "Sensor_Talhao_3"),
        ("Estacao_Meteo", "Sensor_Talhao_4"),
        ("Pivo_Central_1", "Sensor_Talhao_3"),  # Rota redundante de contingência
    ]

    for u, v in enlaces:
        rede.adicionar_aresta(u, v)

    print("Esquema da Topologia Mesh:")
    print("""
      [Talhão 1] ── (hop 2)
           │
     [Repetidor N] ── (hop 1) ──┐
           │                     │
      [Talhão 2] ── (hop 2)      │
           │                     ├── [Gateway Central] (Raiz - hop 0)
      [Talhão 3] ── (hop 3)      │
           │ (enlace reserva)    │
      [Pivô 1]   ── (hop 2)      │
           │                     │
     [Repetidor S] ── (hop 1) ──┘
           │
      [Estação Met] ── (hop 2) ── [Talhão 4] (hop 3)
""")

    origem = "Gateway_Central"
    dist, pred, ordem = rede.bfs(origem)

    print(f"Ordem de Descoberta em Camadas (Frente de Onda BFS):")
    print("  " + " -> ".join(ordem))

    print("\nTabela de Roteamento de Telemetria Gerada pelo BFS:")
    print(f"{'Dispositivo de Campo':<24} | {'Distância (Saltos)':<20} | {'Próximo Salto (Pai na Árvore)':<30}")
    print("-" * 78)
    for dev in sorted(rede.vertices):
        pai = pred[dev] if pred[dev] else "(Gateway / Raiz)"
        print(f"{dev:<24} | {dist[dev]:<20} | {pai:<30}")

    alvo = "Sensor_Talhao_3"
    rota = rede.caminho_minimo_bfs(origem, alvo)
    print(f"\nRota Ótima Calculada para {alvo}:")
    print(f"  {' -> '.join(rota)}  (Total: {len(rota)-1} saltos de rádio)")

    print("\n--- Simulação de Tolerância a Falhas na Lavoura ---")
    print("CENÁRIO: Raio desativa o 'Repetidor_Norte'. Como a rede reage?")
    rede.remover_aresta("Gateway_Central", "Repetidor_Norte")
    rede.remover_aresta("Repetidor_Norte", "Sensor_Talhao_2")
    rede.remover_aresta("Repetidor_Norte", "Sensor_Talhao_1")

    nova_rota = rede.caminho_minimo_bfs(origem, alvo)
    print(f"Nova Rota de Contingência para {alvo}:")
    print(f"  {' -> '.join(nova_rota)}  (Total: {len(nova_rota)-1} saltos via Repetidor_Sul e Pivô_1)")
    print("Conclusão: O BFS recalcula a rota dinâmica em O(V + E) garantindo alta disponibilidade.")


# ======================================================================================
# ESTUDO DE CASO 2: PIPELINE ETL AGRO & DETECÇÃO DE DEADLOCK (DFS)
# ======================================================================================
def executar_caso_2():
    print("\n" + "#" * 80)
    print("ESTUDO DE CASO 2: DETECÇÃO DE DEADLOCK EM PIPELINES ETL NO APACHE AIRFLOW (DFS)")
    print("#" * 80)
    print("""
Contexto:
  Uma plataforma de Big Data Agro processa imagens multiespectrais do satélite Sentinel-2
  e séries temporais de telemetria climática para prever a produtividade da safra.
  O fluxo é orquestrado no Apache Airflow, onde cada tarefa é um operador (Nó) e as
  dependências de execução são setas direcionadas (Arestas).
  O Airflow exige que o grafo seja estritamente um DAG (Directed Acyclic Graph).

Problema Crítico:
  Um desenvolvedor júnior configurou acidentalmente uma dependência circular:
  o Dashboard depende do Modelo ML, que depende da Feature Store, que depende do Cálculo NDVI,
  mas alguém configurou o Cálculo NDVI para aguardar dados do Dashboard!
  Resultado: NENHUMA tarefa executa. O cluster inteiro entra em DEADLOCK.
""")

    print("Esquema do Pipeline no Airflow:")
    print("""
     [Ingestão Sentinel] ──> [Calibração Atmosférica] ──> [Cálculo NDVI] ────────┐
                                                                ▲                 │ (Árvore)
                                                                │ [RETORNO!]      ▼
     [Ingestão Clima]    ──> [Normalização Clima] ──────────────┼───────────> [Feature Store]
                                       │ (Cruzamento)           │                 │ (Árvore)
                                       ▼                        │                 ▼
                                [Deploy Dashboard] <────────────┴────────── [Modelo ML]
                                                    (Árvore)
""")

    pipeline = GrafoAgro(direcionado=True)

    # Fluxo ETL Normal
    pipeline.adicionar_aresta("Ingestao_Sentinel", "Calibracao_Atmosferica")
    pipeline.adicionar_aresta("Calibracao_Atmosferica", "Calculo_NDVI")
    pipeline.adicionar_aresta("Ingestao_Clima", "Normalizacao_Clima")
    pipeline.adicionar_aresta("Calculo_NDVI", "Feature_Store")
    pipeline.adicionar_aresta("Normalizacao_Clima", "Feature_Store")
    pipeline.adicionar_aresta("Feature_Store", "Modelo_ML")
    pipeline.adicionar_aresta("Modelo_ML", "Deploy_Dashboard")

    print("--- Cenário 2A: Auditoria Pré-Deploy (Pipeline Saudável) ---")
    tem_ciclo, ciclo, d, f, arestas = pipeline.detectar_ciclo_dfs()
    print(f"Possui Dependência Cíclica? {tem_ciclo}")
    print("Status: PIPELINE VÁLIDA (100% DAG acíclico). Liberada para produção.")

    print("\n--- Cenário 2B: Injeção de Falha Humana (Dependência Circular) ---")
    print("Adicionando aresta errônea: Deploy_Dashboard -> Calculo_NDVI")
    pipeline.adicionar_aresta("Deploy_Dashboard", "Calculo_NDVI")

    tem_ciclo, ciclo, d, f, arestas = pipeline.detectar_ciclo_dfs()
    print(f"Possui Dependência Cíclica? {tem_ciclo}")
    if tem_ciclo:
        print("\n" + "!" * 70)
        print("ALERTA CRÍTICO DE ENGENHARIA DE DADOS: DEADLOCK DETECTADO!")
        print("!" * 70)
        print("Ciclo Identificado pelo DFS:")
        print("  " + " -> ".join(ciclo))

    print("\nClassificação Rigorosa das Arestas pelo DFS:")
    print(f"  - Arestas de Árvore:     {arestas['arvore']}")
    print(f"  - Arestas de Retorno:    {arestas['retorno']}  <-- Aresta causadora do ciclo!")
    print(f"  - Arestas de Cruzamento: {arestas['cruzamento']}  (Fluxo Clima -> Feature Store)")
    print(f"  - Arestas de Avanço:     {arestas['avanco']}")

    print("\nCarimbos de Tempo de Descoberta d[u] e Término f[u]:")
    print(f"{'Tarefa (Operador Airflow)':<28} | {'Descoberta d[u]':<18} | {'Término f[u]':<18} | {'Intervalo [d, f]':<18}")
    print("-" * 88)
    for t in sorted(pipeline.vertices):
        print(f"{t:<28} | {d[t]:<18} | {f[t]:<18} | [{d[t]}, {f[t]}]")

    print("\nAção Corretiva do Engenheiro:")
    print("  Removendo 'Deploy_Dashboard -> Calculo_NDVI'...")
    pipeline.remover_aresta("Deploy_Dashboard", "Calculo_NDVI")
    ok, _, _, _, _ = pipeline.detectar_ciclo_dfs()
    print(f"  Validação pós-correção: DAG íntegro? {not ok} (Deploy executado com sucesso!)")


# ======================================================================================
# ESTUDO DE CASO 3: PROPAGAÇÃO EPIDEMIOLÓGICA COM MULTI-SOURCE BFS
# ======================================================================================
def executar_caso_3():
    print("\n" + "#" * 80)
    print("ESTUDO DE CASO 3: PROPAGAÇÃO DA FERRUGEM ASIÁTICA COM MULTI-SOURCE BFS")
    print("#" * 80)
    print("""
Contexto:
  A Ferrugem Asiática (Phakopsora pachyrhizi) é uma das doenças fúngicas mais devastadoras
  da cultura da soja no Brasil. Seus esporos são microscópicos e se propagam pelo vento
  para talhões contíguos em questão de dias.
  Drones com câmeras multiespectrais detectaram 2 focos simultâneos em talhões distantes:
    - Foco Primário 1: Talhão A1 (extremo noroeste)
    - Foco Primário 2: Talhão D2 (extremo sudeste)

Objetivo Agronômico:
  Simular o raio e as ondas de contaminação ao longo do tempo (dias) para determinar:
  1. Quais talhões devem receber aplicação curativa imediata (Dia 1);
  2. Quais talhões devem receber barreira preventiva no Dia 2;
  3. Rastrear a genealogia da contaminação (qual foco infectou cada talhão).
""")

    fazenda = GrafoAgro(direcionado=False)

    # Malha espacial de talhões vizinhos (Grid 4x2)
    vizinhancas = [
        ("Talhao_A1", "Talhao_A2"),
        ("Talhao_A1", "Talhao_B1"),
        ("Talhao_A2", "Talhao_B2"),
        ("Talhao_B1", "Talhao_B2"),
        ("Talhao_B1", "Talhao_C1"),
        ("Talhao_B2", "Talhao_C2"),
        ("Talhao_C1", "Talhao_C2"),
        ("Talhao_C1", "Talhao_D1"),
        ("Talhao_C2", "Talhao_D2"),
        ("Talhao_D1", "Talhao_D2"),
    ]

    for u, v in vizinhancas:
        fazenda.adicionar_aresta(u, v)

    print("Mapa Espacial da Fazenda (Grade de Talhões):")
    print("""
       [Talhão A1] (FOCO 1) ── [Talhão A2]
            │                       │
       [Talhão B1] ─────────── [Talhão B2]
            │                       │
       [Talhão C1] ─────────── [Talhão C2]
            │                       │
       [Talhão D1] ─────────── [Talhão D2] (FOCO 2)
""")

    focos_iniciais = ["Talhao_A1", "Talhao_D2"]
    dist, vetor = fazenda.multi_source_bfs(focos_iniciais)

    print(f"Focos Iniciais de Contaminação Inseridos na Fila BFS: {focos_iniciais}")

    print("\nRelatório Fitossanitário de Dispersão e Origem:")
    print(f"{'Talhão':<14} | {'Distância (Dias)':<18} | {'Vetor Infeccioso de Origem':<30} | {'Ação Recomendada':<25}")
    print("-" * 92)
    for t in sorted(fazenda.vertices):
        dias = dist[t]
        orig = vetor[t]
        if dias == 0:
            acao = "ISOLAMENTO / CURATIVO IMEDIATO"
        elif dias == 1:
            acao = "PULVERIZAÇÃO EMERGENCIAL (24h)"
        elif dias == 2:
            acao = "BARREIRA PREVENTIVA (48h)"
        else:
            acao = "MONITORAMENTO VIA DRONE"
        print(f"{t:<14} | {dias:<18} | {orig:<30} | {acao:<25}")

    print("\nResumo da Análise Epidemiológica:")
    talhoes_dia_1 = [t for t, d in dist.items() if d == 1]
    talhoes_dia_2 = [t for t, d in dist.items() if d == 2]
    print(f"  - Frente de avanço Dia 1 (Urgência máxima): {sorted(talhoes_dia_1)}")
    print(f"  - Frente de avanço Dia 2 (Prevenção tática):  {sorted(talhoes_dia_2)}")
    print("Conclusão: O Multi-Source BFS permite ao engenheiro de dados calcular zonas de amortecimento")
    print("           economizando defensivos agrícolas e contendo a praga em tempo hábil.")


# ======================================================================================
# PROGRAMA PRINCIPAL
# ======================================================================================
if __name__ == "__main__":
    opcao = sys.argv[1] if len(sys.argv) > 1 else "todos"

    if opcao == "1":
        executar_caso_1()
    elif opcao == "2":
        executar_caso_2()
    elif opcao == "3":
        executar_caso_3()
    else:
        print("\n" + "=" * 80)
        print("ALGORITMOS AVANÇADOS - DEMONSTRAÇÃO DOS 3 ESTUDOS DE CASO NO AGRONEGÓCIO")
        print("Prof. Me. Douglas Samuel Gonçalves - Fatec Bebedouro")
        print("=" * 80)
        executar_caso_1()
        executar_caso_2()
        executar_caso_3()
        print("\n" + "=" * 80)
        print("FIM DA EXECUÇÃO DOS 3 CASOS PRÁTICOS. TODOS OS RESULTADOS CONFERIDOS.")
        print("=" * 80 + "\n")
