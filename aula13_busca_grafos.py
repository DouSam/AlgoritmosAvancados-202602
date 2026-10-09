"""
Algoritmos Avançados - Aula 17: Algoritmos de Busca em Grafos (BFS e DFS)
Prof. Me. Douglas Samuel Gonçalves
Fatec Bebedouro - Jorge Caram Sabbag
Tecnologia em Big Data no Agronegócio (2º Semestre / 2026)

Este módulo implementa a classe GrafoBusca, estendendo o TAD Grafo com:
- Busca em Largura (BFS - Breadth-First Search) e menor caminho não-ponderado
- Busca Multi-Fonte (Multi-Source BFS) para dispersão epidemiológica e telemetria
- Busca em Profundidade (DFS - Depth-First Search) recursiva e iterativa
- Marcação de tempos de descoberta (d) e finalização (f)
- Detecção de ciclos em grafos direcionados e não-direcionados com reconstrução do ciclo
- Classificação de arestas (Árvore, Retorno, Avanço, Cruzamento)
- Componentes conexos
- Estudos de caso reais em Big Data e Agronegócio
"""

import sys
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Configura saída padrão UTF-8 para evitar caracteres corrompidos em terminais Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class CorDFS(Enum):
    """Estados dos vértices durante a execução da Busca em Profundidade."""
    BRANCO = 0  # Não visitado / Não descoberto
    CINZA = 1   # Descoberto, atualmente na pilha de recursão / em processamento
    PRETO = 2   # Totalmente explorado, todos os vizinhos visitados


class Grafo:
    """
    Representação de Grafo utilizando Dicionário de Adjacência.
    Suporta grafos direcionados, não-direcionados e ponderados.
    Base consolidada na Aula 16.
    """

    def __init__(self, direcionado: bool = False):
        self._direcionado: bool = direcionado
        # Estrutura: {vertice_origem: {vertice_destino: peso}}
        self._adj: Dict[str, Dict[str, float]] = {}

    @property
    def eh_direcionado(self) -> bool:
        """Indica se o grafo é direcionado."""
        return self._direcionado

    @property
    def total_vertices(self) -> int:
        """Retorna o número total de vértices (|V|)."""
        return len(self._adj)

    @property
    def vertices(self) -> List[str]:
        """Retorna a lista de identificadores de todos os vértices."""
        return list(self._adj.keys())

    def adicionar_vertice(self, vertice: str) -> bool:
        """Adiciona um novo vértice isolado caso não exista."""
        if vertice not in self._adj:
            self._adj[vertice] = {}
            return True
        return False

    def adicionar_aresta(self, u: str, v: str, peso: float = 1.0) -> None:
        """Cria uma aresta ponderada entre u e v."""
        self.adicionar_vertice(u)
        self.adicionar_vertice(v)

        self._adj[u][v] = peso
        if not self._direcionado:
            self._adj[v][u] = peso

    def remover_aresta(self, u: str, v: str) -> bool:
        """Remove a aresta entre u e v. Retorna True se removida."""
        if u in self._adj and v in self._adj[u]:
            del self._adj[u][v]
            if not self._direcionado and v in self._adj and u in self._adj[v]:
                del self._adj[v][u]
            return True
        return False

    def obter_vizinhos(self, vertice: str) -> List[str]:
        """Retorna a lista de vizinhos adjacentes ao vértice em ordem alfabética estável."""
        if vertice in self._adj:
            return sorted(self._adj[vertice].keys())
        return []

    def obter_peso(self, u: str, v: str) -> Optional[float]:
        """Retorna o peso da aresta (u, v) ou None se inexistente."""
        return self._adj.get(u, {}).get(v, None)

    def grau(self, vertice: str) -> int:
        """Retorna o grau de saída ou grau total do vértice."""
        if vertice not in self._adj:
            raise KeyError(f"Vértice '{vertice}' não encontrado no grafo.")
        return len(self._adj[vertice])

    def total_arestas(self) -> int:
        """Calcula o número total de arestas no grafo."""
        total = sum(len(vizinhos) for vizinhos in self._adj.values())
        return total if self._direcionado else total // 2

    def __str__(self) -> str:
        tipo = "Direcionado" if self._direcionado else "Não-Direcionado"
        linhas = [f"=== Grafo ({tipo} | |V|={self.total_vertices}, |E|={self.total_arestas()}) ==="]
        for v in sorted(self._adj.keys()):
            conexoes = ", ".join(f"{dest}(w={peso:.1f})" for dest, peso in sorted(self._adj[v].items()))
            linhas.append(f"  [{v}] -> {conexoes if conexoes else 'Nenhum vizinho'}")
        return "\n".join(linhas)


class GrafoBusca(Grafo):
    """
    Extensão especializada do Grafo com algoritmos de travessia e busca:
    BFS, DFS, Menor Caminho em saltos, Detecção de Ciclos e Componentes Conexos.
    """

    def bfs(self, inicio: str) -> Dict[str, Any]:
        """
        Executa a Busca em Largura (BFS) a partir de um vértice inicial.
        Complexidade de Tempo: O(V + E)
        Complexidade de Espaço: O(V)
        
        Retorna dicionário contendo:
        - 'ordem': lista de nós na ordem de visitação
        - 'distancias': dict {vertice: distancia_em_saltos} (ou infinito se inalcançável)
        - 'predecessores': dict {vertice: pai_na_arvore_bfs}
        - 'niveis': dict {nivel_int: [vertices]}
        """
        if inicio not in self._adj:
            raise KeyError(f"Vértice de início '{inicio}' não existe no grafo.")

        visitados: Set[str] = {inicio}
        distancias: Dict[str, float] = {v: float("inf") for v in self.vertices}
        predecessores: Dict[str, Optional[str]] = {v: None for v in self.vertices}
        ordem_visita: List[str] = []
        niveis: Dict[int, List[str]] = {}

        distancias[inicio] = 0
        fila: deque[str] = deque([inicio])

        while fila:
            u = fila.popleft()
            ordem_visita.append(u)
            d_atual = int(distancias[u])
            niveis.setdefault(d_atual, []).append(u)

            for v in self.obter_vizinhos(u):
                if v not in visitados:
                    visitados.add(v)
                    distancias[v] = d_atual + 1
                    predecessores[v] = u
                    fila.append(v)

        return {
            "origem": inicio,
            "ordem": ordem_visita,
            "distancias": {v: (int(d) if d != float("inf") else -1) for v, d in distancias.items()},
            "predecessores": predecessores,
            "niveis": niveis,
        }

    def caminho_minimo_bfs(self, inicio: str, destino: str) -> List[str]:
        """
        Retorna o caminho de menor número de arestas (saltos) entre início e destino.
        Se não houver caminho, retorna lista vazia.
        """
        if inicio not in self._adj or destino not in self._adj:
            return []

        resultado = self.bfs(inicio)
        predecessores = resultado["predecessores"]

        if resultado["distancias"].get(destino, -1) == -1:
            return []  # Destino inalcançável

        # Reconstrói o caminho de trás para frente a partir do destino
        caminho: List[str] = []
        atual: Optional[str] = destino
        while atual is not None:
            caminho.append(atual)
            if atual == inicio:
                break
            atual = predecessores[atual]

        caminho.reverse()
        return caminho

    def multi_source_bfs(self, fontes: List[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Busca em Largura com múltiplas fontes simultâneas.
        Útil para simular propagação epidemiológica (pragas) ou roteamento de sensores
        para múltiplos gateways.
        
        Retorna:
        - distancias: dict {vertice: menor_distancia_ate_alguma_fonte}
        - fonte_mais_proxima: dict {vertice: qual_fonte_o_alcancou_primeiro}
        """
        distancias: Dict[str, float] = {v: float("inf") for v in self.vertices}
        fonte_mais_proxima: Dict[str, Optional[str]] = {v: None for v in self.vertices}
        fila: deque[Tuple[str, str]] = deque()  # (nó, fonte_origem)

        fontes_unicas = list(dict.fromkeys(fontes))
        for f in fontes_unicas:
            if f in self._adj:
                distancias[f] = 0
                fonte_mais_proxima[f] = f
                fila.append((f, f))

        while fila:
            u, f_origem = fila.popleft()
            d_u = int(distancias[u])

            for v in self.obter_vizinhos(u):
                if distancias[v] == float("inf"):
                    distancias[v] = d_u + 1
                    fonte_mais_proxima[v] = f_origem
                    fila.append((v, f_origem))

        res_dist = {v: (int(d) if d != float("inf") else -1) for v, d in distancias.items()}
        res_fonte = {v: (fonte_mais_proxima[v] or "N/A") for v in self.vertices}
        return res_dist, res_fonte

    def dfs_recursivo(self, inicio: Optional[str] = None) -> Dict[str, Any]:
        """
        Executa a Busca em Profundidade (DFS) com registro de tempos de descoberta
        (d[v]) e finalização (f[v]), cores e predecessores.
        
        Se 'inicio' for fornecido, executa DFS a partir dele e depois nos demais
        vértices não visitados para cobrir a floresta completa em ordem canônica.
        """
        cores: Dict[str, CorDFS] = {v: CorDFS.BRANCO for v in self.vertices}
        tempo_d: Dict[str, int] = {}
        tempo_f: Dict[str, int] = {}
        predecessores: Dict[str, Optional[str]] = {v: None for v in self.vertices}
        ordem_visita: List[str] = []
        tempo: int = 0

        def visitar(u: str):
            nonlocal tempo
            tempo += 1
            tempo_d[u] = tempo
            cores[u] = CorDFS.CINZA
            ordem_visita.append(u)

            for v in self.obter_vizinhos(u):
                if cores[v] == CorDFS.BRANCO:
                    predecessores[v] = u
                    visitar(v)

            cores[u] = CorDFS.PRETO
            tempo += 1
            tempo_f[u] = tempo

        # Ordem de início prioritária canônica
        vertices_ordenados = sorted(self.vertices)
        if inicio and inicio in vertices_ordenados:
            vertices_ordenados.remove(inicio)
            vertices_ordenados.insert(0, inicio)

        for vertice in vertices_ordenados:
            if cores[vertice] == CorDFS.BRANCO:
                visitar(vertice)

        return {
            "ordem": ordem_visita,
            "tempo_descoberta": tempo_d,
            "tempo_finalizacao": tempo_f,
            "predecessores": predecessores,
        }

    def dfs_iterativo(self, inicio: str) -> List[str]:
        """
        Implementação do DFS utilizando uma Pilha explícita (LIFO).
        Previne estouro da pilha de recursão (RecursionError) em grafos massivos.
        """
        if inicio not in self._adj:
            return []

        visitados: Set[str] = set()
        ordem_visita: List[str] = []
        pilha: List[str] = [inicio]

        while pilha:
            u = pilha.pop()
            if u not in visitados:
                visitados.add(u)
                ordem_visita.append(u)
                # Invertemos os vizinhos para que o menor em ordem alfabética
                # fique no topo da pilha e seja desempilhado primeiro
                for v in reversed(self.obter_vizinhos(u)):
                    if v not in visitados:
                        pilha.append(v)

        return ordem_visita

    def tem_ciclo_direcionado(self) -> Tuple[bool, List[str]]:
        """
        Detecta ciclos em grafo direcionado utilizando 3 cores (BRANCO, CINZA, PRETO).
        Uma aresta para um nó CINZA é uma Aresta de Retorno (Back Edge), provando a
        existência de ciclo.
        
        Retorna:
        - (True, [caminho_do_ciclo]) se existir ciclo
        - (False, []) se o grafo for um DAG (Grafo Acíclico Direcionado)
        """
        if not self._direcionado:
            raise ValueError("Este método é aplicável apenas a grafos direcionados.")

        cores: Dict[str, CorDFS] = {v: CorDFS.BRANCO for v in self.vertices}
        pai: Dict[str, Optional[str]] = {v: None for v in self.vertices}
        ciclo_encontrado: List[str] = []

        def dfs_ciclo(u: str) -> bool:
            cores[u] = CorDFS.CINZA

            for v in self.obter_vizinhos(u):
                if cores[v] == CorDFS.CINZA:
                    # Ciclo detectado! Reconstrói o caminho do ciclo
                    ciclo_encontrado.append(v)
                    curr = u
                    while curr != v and curr is not None:
                        ciclo_encontrado.append(curr)
                        curr = pai[curr]
                    ciclo_encontrado.append(v)
                    ciclo_encontrado.reverse()
                    return True
                elif cores[v] == CorDFS.BRANCO:
                    pai[v] = u
                    if dfs_ciclo(v):
                        return True

            cores[u] = CorDFS.PRETO
            return False

        for vertice in sorted(self.vertices):
            if cores[vertice] == CorDFS.BRANCO:
                if dfs_ciclo(vertice):
                    return True, ciclo_encontrado

        return False, []

    def tem_ciclo_nao_direcionado(self) -> Tuple[bool, List[str]]:
        """
        Detecta ciclos em grafo não-direcionado.
        Um ciclo existe se encontramos um vizinho já visitado que NÃO seja o pai imediato.
        """
        if self._direcionado:
            raise ValueError("Este método é aplicável apenas a grafos não-direcionados.")

        visitados: Set[str] = set()
        pai: Dict[str, Optional[str]] = {v: None for v in self.vertices}
        ciclo: List[str] = []

        def dfs_ciclo_nd(u: str) -> bool:
            visitados.add(u)
            for v in self.obter_vizinhos(u):
                if v not in visitados:
                    pai[v] = u
                    if dfs_ciclo_nd(v):
                        return True
                elif v != pai[u]:
                    # Vizinho já visitado que não é o pai imediato -> ciclo detectado!
                    ciclo.append(v)
                    curr = u
                    while curr != v and curr is not None:
                        ciclo.append(curr)
                        curr = pai[curr]
                    ciclo.append(v)
                    ciclo.reverse()
                    return True
            return False

        for vertice in sorted(self.vertices):
            if vertice not in visitados:
                if dfs_ciclo_nd(vertice):
                    return True, ciclo

        return False, []

    def classificar_arestas_dfs(self, inicio: Optional[str] = None) -> Dict[str, List[Tuple[str, str]]]:
        """
        Classifica todas as arestas do grafo conforme a travessia DFS:
        - Árvore (Tree Edge): aresta que descobre um novo vértice (BRANCO).
        - Retorno (Back Edge): aresta conectando a um ancestral (CINZA). Indica ciclo!
        - Avanço (Forward Edge): aresta para um descendente já finalizado (PRETO, d[u] < d[v]).
        - Cruzamento (Cross Edge): aresta entre ramos disjuntos (PRETO, d[u] > d[v]).
        """
        if not self._direcionado:
            raise ValueError("A classificação em 4 tipos é formalmente definida para digrafos.")

        dfs_res = self.dfs_recursivo(inicio)
        d = dfs_res["tempo_descoberta"]
        f = dfs_res["tempo_finalizacao"]
        preds = dfs_res["predecessores"]

        classificacao: Dict[str, List[Tuple[str, str]]] = {
            "arvore": [],
            "retorno": [],
            "avanco": [],
            "cruzamento": [],
        }

        for u in sorted(self.vertices):
            for v in self.obter_vizinhos(u):
                if preds[v] == u:
                    classificacao["arvore"].append((u, v))
                elif d[v] <= d[u] and f[u] <= f[v]:
                    classificacao["retorno"].append((u, v))
                elif d[u] < d[v] and f[v] < f[u]:
                    classificacao["avanco"].append((u, v))
                else:
                    classificacao["cruzamento"].append((u, v))

        return classificacao

    def componentes_conectados(self) -> List[List[str]]:
        """
        Determina todas as componentes conexas de um grafo não-direcionado.
        Retorna lista de listas de vértices.
        """
        if self._direcionado:
            raise ValueError(
                "Componentes conexas são formalmente definidas para grafos não-direcionados. "
                "Para digrafos, consulte Componentes Fortemente Conexas (SCC) na Aula 18."
            )

        visitados: Set[str] = set()
        componentes: List[List[str]] = []

        for v in sorted(self.vertices):
            if v not in visitados:
                comp: List[str] = []
                fila = deque([v])
                visitados.add(v)

                while fila:
                    curr = fila.popleft()
                    comp.append(curr)
                    for viz in self.obter_vizinhos(curr):
                        if viz not in visitados:
                            visitados.add(viz)
                            fila.append(viz)

                componentes.append(sorted(comp))

        return componentes


# ====================================================================
# Estudos de Caso Práticos e Demonstrações da Aula 17
# ====================================================================

def executar_estudo_caso_1_mesh_telemetria() -> None:
    """
    Caso 1: Roteamento de Telemetria Agrícola em Rede Mesh LoRaWAN
    Nós: Sensores de Solo, Estações Meteorológicas, Pivôs Centrais e Gateway Central.
    Objetivo: Identificar o menor número de saltos até a Usina (Gateway) e árvore de espalhamento.
    """
    print("\n" + "=" * 70)
    print("ESTUDO DE CASO 1: REDE MESH DE TELEMETRIA AGRÍCOLA (BFS)")
    print("=" * 70)

    rede = GrafoBusca(direcionado=False)
    # Adicionando enlaces sem fio entre sensores nos talhões
    rede.adicionar_aresta("Gateway_Central", "Repetidor_Norte")
    rede.adicionar_aresta("Gateway_Central", "Repetidor_Sul")
    rede.adicionar_aresta("Repetidor_Norte", "Sensor_Talhao_1")
    rede.adicionar_aresta("Repetidor_Norte", "Sensor_Talhao_2")
    rede.adicionar_aresta("Repetidor_Sul", "Pivo_Central_1")
    rede.adicionar_aresta("Repetidor_Sul", "Estacao_Meteo")
    rede.adicionar_aresta("Sensor_Talhao_2", "Sensor_Talhao_3")
    rede.adicionar_aresta("Pivo_Central_1", "Sensor_Talhao_3")
    rede.adicionar_aresta("Estacao_Meteo", "Sensor_Talhao_4")

    print(f"Topologia da Rede: {rede.total_vertices} dispositivos, {rede.total_arestas()} enlaces de rádio.")

    # Executa BFS a partir do Gateway Central
    resultado_bfs = rede.bfs("Gateway_Central")
    print(f"\nOrdem de Descoberta (Onda BFS): {' -> '.join(resultado_bfs['ordem'])}")

    print("\nTabela de Roteamento (Distância em Saltos e Próximo Salto):")
    print(f"{'Dispositivo':<22} | {'Saltos (hops)':<13} | {'Antecessor na Árvore':<20}")
    print("-" * 62)
    for dev in sorted(rede.vertices):
        dist = resultado_bfs["distancias"][dev]
        pai = resultado_bfs["predecessores"][dev] or "(Origem / Raiz)"
        print(f"{dev:<22} | {dist:<13} | {pai:<20}")

    # Reconstrução de Rota
    alvo = "Sensor_Talhao_3"
    rota = rede.caminho_minimo_bfs("Gateway_Central", alvo)
    print(f"\nRota Ótima (Menor Salto) até {alvo}:")
    print("  " + " -> ".join(rota) + f" (Total de Saltos: {len(rota) - 1})")


def executar_estudo_caso_2_dag_pipeline() -> None:
    """
    Caso 2: Pipeline de Engenharia de Dados Agro no Apache Airflow (DFS)
    Tarefas de ETL para processamento de imagens de satélite Sentinel e dados de colheita.
    Objetivo: Validar ausência de ciclos (garantia de DAG) e detectar dependência circular.
    """
    print("\n" + "=" * 70)
    print("ESTUDO DE CASO 2: PIPELINE DE DADOS AGRO & DETECÇÃO DE DEADLOCK (DFS)")
    print("=" * 70)

    pipeline = GrafoBusca(direcionado=True)

    # Tarefas da Pipeline ETL
    pipeline.adicionar_aresta("Ingestao_Sentinel", "Calibracao_Atmosferica")
    pipeline.adicionar_aresta("Calibracao_Atmosferica", "Calculo_NDVI")
    pipeline.adicionar_aresta("Ingestao_Clima", "Normalizacao_Clima")
    pipeline.adicionar_aresta("Calculo_NDVI", "Feature_Store")
    pipeline.adicionar_aresta("Normalizacao_Clima", "Feature_Store")
    pipeline.adicionar_aresta("Feature_Store", "Treinamento_Modelo_Safra")
    pipeline.adicionar_aresta("Treinamento_Modelo_Safra", "Deploy_Dashboard")

    print("--- Cenário 2A: Pipeline Válida (DAG Acíclico) ---")
    tem_ciclo, caminho_ciclo = pipeline.tem_ciclo_direcionado()
    print(f"Possui Ciclo? {tem_ciclo}")
    if not tem_ciclo:
        print("  Status: PIPELINE APROVADA PARA EXECUÇÃO NO AIRFLOW (DAG Acíclico).")

    # DFS e tempos d/f
    dfs_res = pipeline.dfs_recursivo("Ingestao_Sentinel")
    print("\nTempos de Descoberta (d) e Finalização (f) na Pipeline:")
    print(f"{'Tarefa':<28} | {'Descoberta (d)':<15} | {'Finalização (f)':<15}")
    print("-" * 64)
    for task in sorted(pipeline.vertices):
        d_val = dfs_res["tempo_descoberta"].get(task, "-")
        f_val = dfs_res["tempo_finalizacao"].get(task, "-")
        print(f"{task:<28} | {d_val:<15} | {f_val:<15}")

    print("\n--- Cenário 2B: Injeção de Erro de Dependência Cíclica ---")
    # Adicionando uma aresta inválida que causa ciclo: Deploy_Dashboard depende de volta de Calculo_NDVI
    pipeline.adicionar_aresta("Deploy_Dashboard", "Calculo_NDVI")
    print("Aresta Adicionada: Deploy_Dashboard -> Calculo_NDVI (Dependência Circular)")

    tem_ciclo, caminho_ciclo = pipeline.tem_ciclo_direcionado()
    print(f"Possui Ciclo? {tem_ciclo}")
    if tem_ciclo:
        print(f"  ALERTA CRÍTICO: Deadlock detectado na pipeline!")
        print(f"  Ciclo identificado: {' -> '.join(caminho_ciclo)}")

    classif = pipeline.classificar_arestas_dfs("Ingestao_Sentinel")
    print("\nClassificação das Arestas no Digrafo com Ciclo:")
    print(f"  - Arestas de Árvore:    {classif['arvore']}")
    print(f"  - Arestas de Retorno:   {classif['retorno']}  <-- Causadora do Ciclo!")
    print(f"  - Arestas de Avanço:    {classif['avanco']}")
    print(f"  - Arestas de Cruzamento:{classif['cruzamento']}")


def executar_estudo_caso_3_epidemiologia() -> None:
    """
    Caso 3: Dispersão Epidemiológica (Ferrugem Asiática) com Multi-Source BFS
    Múltiplos focos de infecção inicial nos talhões propagando-se por proximidade.
    """
    print("\n" + "=" * 70)
    print("ESTUDO DE CASO 3: PROPAGAÇÃO EPIDEMIOLÓGICA MULTI-FONTE (FERRUGEM ASIÁTICA)")
    print("=" * 70)

    fazenda = GrafoBusca(direcionado=False)
    # Malha de talhões vizinhos
    fazenda.adicionar_aresta("Talhao_A1", "Talhao_A2")
    fazenda.adicionar_aresta("Talhao_A1", "Talhao_B1")
    fazenda.adicionar_aresta("Talhao_A2", "Talhao_B2")
    fazenda.adicionar_aresta("Talhao_B1", "Talhao_B2")
    fazenda.adicionar_aresta("Talhao_B2", "Talhao_C2")
    fazenda.adicionar_aresta("Talhao_B1", "Talhao_C1")
    fazenda.adicionar_aresta("Talhao_C1", "Talhao_C2")
    fazenda.adicionar_aresta("Talhao_C2", "Talhao_D2")
    fazenda.adicionar_aresta("Talhao_D1", "Talhao_D2")
    fazenda.adicionar_aresta("Talhao_C1", "Talhao_D1")

    focos_iniciais = ["Talhao_A1", "Talhao_D2"]
    print(f"Focos Iniciais de Contaminação Detectados: {focos_iniciais}")

    dist, fonte = fazenda.multi_source_bfs(focos_iniciais)

    print("\nOndas de Dispersão e Raio de Contaminação:")
    print(f"{'Talhão':<14} | {'Distância (Dias/Ondas)':<25} | {'Vetor de Origem Mais Próximo':<30}")
    print("-" * 75)
    for t in sorted(fazenda.vertices):
        print(f"{t:<14} | {dist[t]:<25} | {fonte[t]:<30}")


def executar_exercicio_1_bioinsumos() -> None:
    """
    Exercício 1 dos Slides: Distribuição de Bioinsumos entre Cooperativas (BFS)
    Vértices: Matriz (M), Cooperativas (C1, C2), Produção (P1, P2), Distribuição (D).
    """
    print("\n" + "=" * 70)
    print("EXERCÍCIO 1 (SLIDES): REDE DE BIOINSUMOS AGRÍCOLAS (BFS)")
    print("=" * 70)

    rede = GrafoBusca(direcionado=False)
    rede.adicionar_aresta("M", "C1")
    rede.adicionar_aresta("M", "C2")
    rede.adicionar_aresta("C1", "P1")
    rede.adicionar_aresta("C1", "C2")
    rede.adicionar_aresta("C2", "P2")
    rede.adicionar_aresta("P1", "D")
    rede.adicionar_aresta("P2", "D")
    rede.adicionar_aresta("P1", "P2")

    resultado = rede.bfs("M")
    print(f"Ordem de Descoberta BFS a partir da Matriz (M): {' -> '.join(resultado['ordem'])}")

    print("\nTabela de Distâncias e Predecessores (Gabarito Oficial):")
    print(f"{'Cooperativa':<14} | {'Distância (d)':<15} | {'Predecessor (pi)':<18}")
    print("-" * 52)
    for v in sorted(rede.vertices):
        d_val = resultado["distancias"][v]
        p_val = resultado["predecessores"][v] or "-"
        print(f"{v:<14} | {d_val:<15} | {p_val:<18}")

    rota_d = rede.caminho_minimo_bfs("M", "D")
    print(f"\nMenor Rota de M até D: {' -> '.join(rota_d)} (Distância: {len(rota_d) - 1} saltos)")


def executar_exercicio_2_satelite_ciclo() -> None:
    """
    Exercício 2 dos Slides: Dependências de Satélite e Detecção de Ciclo (DFS)
    Digrafo: A -> B, B -> C, C -> D, A -> D, D -> B (ciclo em B-C-D-B).
    """
    print("\n" + "=" * 70)
    print("EXERCÍCIO 2 (SLIDES): PIPELINE DE SATÉLITE & ARESTAS DFS")
    print("=" * 70)

    digrafo = GrafoBusca(direcionado=True)
    digrafo.adicionar_aresta("A", "B")
    digrafo.adicionar_aresta("B", "C")
    digrafo.adicionar_aresta("C", "D")
    digrafo.adicionar_aresta("A", "D")
    digrafo.adicionar_aresta("D", "B")

    dfs_res = digrafo.dfs_recursivo("A")
    print("Carimbos de Tempo de Descoberta (d) e Finalização (f):")
    print(f"{'Vértice':<10} | {'Descoberta (d)':<16} | {'Finalização (f)':<16} | {'Intervalo [d, f]':<18}")
    print("-" * 65)
    for v in sorted(digrafo.vertices):
        d_val = dfs_res["tempo_descoberta"][v]
        f_val = dfs_res["tempo_finalizacao"][v]
        print(f"{v:<10} | {d_val:<16} | {f_val:<16} | [{d_val}, {f_val}]")

    classif = digrafo.classificar_arestas_dfs("A")
    print("\nClassificação Rigorosa das Arestas:")
    print(f"  - Arestas de Árvore:    {classif['arvore']}")
    print(f"  - Arestas de Retorno:   {classif['retorno']}  <-- Gera dependência circular!")
    print(f"  - Arestas de Avanço:    {classif['avanco']}")
    print(f"  - Arestas de Cruzamento:{classif['cruzamento']}")

    tem_ciclo, rota_ciclo = digrafo.tem_ciclo_direcionado()
    print(f"\nGrafo é um DAG acíclico? {not tem_ciclo}")
    if tem_ciclo:
        print(f"Ciclo Crítico Detectado: {' -> '.join(rota_ciclo)}")


def executar_verificacao_automatica() -> None:
    """Testes unitários automatizados para garantir integridade matemática."""
    g = GrafoBusca(direcionado=False)
    g.adicionar_aresta("A", "B")
    g.adicionar_aresta("B", "C")
    g.adicionar_aresta("C", "D")
    g.adicionar_aresta("D", "A")
    g.adicionar_aresta("B", "D")

    # Teste BFS Menor Caminho
    caminho = g.caminho_minimo_bfs("A", "C")
    assert len(caminho) == 3, f"Esperado 3 vértices no caminho A-C, obtido {caminho}"
    assert caminho[0] == "A" and caminho[-1] == "C"

    # Teste BFS mesmo início e destino
    assert g.caminho_minimo_bfs("A", "A") == ["A"]

    # Teste Ciclo Não Direcionado
    tem_c, ciclo_nd = g.tem_ciclo_nao_direcionado()
    assert tem_c is True, "O grafo ABCD possui ciclos!"
    assert len(ciclo_nd) >= 4 and ciclo_nd[0] == ciclo_nd[-1]

    # Teste DFS Iterativo vs Recursivo
    ordem_rec = g.dfs_recursivo("A")["ordem"]
    ordem_iter = g.dfs_iterativo("A")
    assert set(ordem_rec) == set(ordem_iter) == {"A", "B", "C", "D"}

    # Teste DAG
    dag = GrafoBusca(direcionado=True)
    dag.adicionar_aresta("1", "2")
    dag.adicionar_aresta("2", "3")
    tem_c_dir, _ = dag.tem_ciclo_direcionado()
    assert tem_c_dir is False, "1->2->3 deve ser acíclico"

    dag.adicionar_aresta("3", "1")
    tem_c_dir2, ciclo_rota = dag.tem_ciclo_direcionado()
    assert tem_c_dir2 is True, "1->2->3->1 deve ter ciclo"
    assert len(ciclo_rota) == 4, f"Rota do ciclo esperada com 4 elementos: {ciclo_rota}"

    # Teste Multi-Source BFS com fontes duplicadas
    dist_dup, fonte_dup = g.multi_source_bfs(["A", "A", "C"])
    assert dist_dup["A"] == 0 and dist_dup["C"] == 0
    assert dist_dup["B"] == 1

    # Teste Componentes Conexas e Validação em Digrafo
    comps = g.componentes_conectados()
    assert len(comps) == 1 and comps[0] == ["A", "B", "C", "D"]

    try:
        dag.componentes_conectados()
        assert False, "Deveria lançar ValueError em grafo direcionado"
    except ValueError:
        pass

    # Teste Exercício 1 (Gabarito dos Slides)
    g1 = GrafoBusca(direcionado=False)
    for u, v in [("M", "C1"), ("M", "C2"), ("C1", "P1"), ("C1", "C2"), ("C2", "P2"), ("P1", "D"), ("P2", "D"), ("P1", "P2")]:
        g1.adicionar_aresta(u, v)
    assert g1.caminho_minimo_bfs("M", "D") == ["M", "C1", "P1", "D"]

    # Teste Exercício 2 (Gabarito dos Slides)
    g2 = GrafoBusca(direcionado=True)
    for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"), ("D", "B")]:
        g2.adicionar_aresta(u, v)
    cl2 = g2.classificar_arestas_dfs("A")
    assert cl2["retorno"] == [("D", "B")]
    assert cl2["avanco"] == [("A", "D")]
    c2_ok, c2_rota = g2.tem_ciclo_direcionado()
    assert c2_ok and c2_rota == ["B", "C", "D", "B"]

    print("\n>>> TODOS OS TESTES UNITÁRIOS DE INTEGRIDADE PASSARAM COM SUCESSO! <<<")


if __name__ == "__main__":
    executar_verificacao_automatica()
    executar_estudo_caso_1_mesh_telemetria()
    executar_estudo_caso_2_dag_pipeline()
    executar_estudo_caso_3_epidemiologia()
    executar_exercicio_1_bioinsumos()
    executar_exercicio_2_satelite_ciclo()
