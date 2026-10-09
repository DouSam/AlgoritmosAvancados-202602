"""
========================================================================================
Algoritmos Avançados - Aulas 27 & 28: Caminho Mínimo em Grafos (BFS e Dijkstra)
Prof. Me. Douglas Samuel Gonçalves
Fatec Bebedouro - Jorge Caram Sabbag
Tecnologia em Big Data no Agronegócio (2º Semestre / 2026)
========================================================================================

Este módulo implementa o TAD GrafoCaminhoMinimo com:
1. Caminho Mínimo em Grafos Não-Ponderados via BFS (O(V + E)).
2. Métricas Topológicas Globais: Excentricidade, Raio, Diâmetro e Centro do Grafo.
3. Demonstração da falha do BFS em grafos ponderados (Contraexemplo Didático).
4. Algoritmo de Dijkstra com Fila de Prioridades Min-Heap (heapq) em O((V + E) log V).
5. Reconstrução de rotas ótimas e cálculo de custo acumulado.
6. Estudos de Caso no Agronegócio:
   - Caso 1: Roteamento Rodoviário Regional de Safra (Bebedouro - Barretos - Ribeirão Preto)
   - Caso 2: Localização Ótima de Hub Logístico de Fertilizantes (Centro do Grafo)
   - Caso 3: Roteamento Multicritério com Pedágio e Custo Diesel
7. Resolução automatizada dos Exercícios de Sala.
========================================================================================
"""

import heapq
import sys
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

# Reconfigura stdout para UTF-8 no Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class Grafo:
    """
    Estrutura base de Grafo utilizando Dicionário de Adjacência.
    Suporta grafos direcionados, não-direcionados e ponderados.
    """

    def __init__(self, direcionado: bool = False):
        self._direcionado: bool = direcionado
        # Estrutura: {u: {v: peso}}
        self._adj: Dict[str, Dict[str, float]] = {}

    @property
    def eh_direcionado(self) -> bool:
        return self._direcionado

    @property
    def total_vertices(self) -> int:
        return len(self._adj)

    @property
    def vertices(self) -> List[str]:
        return sorted(self._adj.keys())

    def adicionar_vertice(self, v: str) -> bool:
        if v not in self._adj:
            self._adj[v] = {}
            return True
        return False

    def adicionar_aresta(self, u: str, v: str, peso: float = 1.0) -> None:
        self.adicionar_vertice(u)
        self.adicionar_vertice(v)
        self._adj[u][v] = float(peso)
        if not self._direcionado:
            self._adj[v][u] = float(peso)

    def obter_vizinhos(self, u: str) -> List[str]:
        if u in self._adj:
            return sorted(self._adj[u].keys())
        return []

    def obter_peso(self, u: str, v: str) -> Optional[float]:
        return self._adj.get(u, {}).get(v, None)

    def total_arestas(self) -> int:
        total = sum(len(viz) for viz in self._adj.values())
        return total if self._direcionado else total // 2

    def __str__(self) -> str:
        tipo = "Direcionado" if self._direcionado else "Não-Direcionado"
        linhas = [f"=== Grafo ({tipo} | |V|={self.total_vertices}, |E|={self.total_arestas()}) ==="]
        for u in sorted(self._adj.keys()):
            viz = ", ".join(f"{v}(w={peso:.1f})" for v, peso in sorted(self._adj[u].items()))
            linhas.append(f"  [{u}] -> {viz if viz else 'Sem conexões'}")
        return "\n".join(linhas)


class GrafoCaminhoMinimo(Grafo):
    """
    Extensão especializada com algoritmos de Caminho Mínimo e Métricas de Rede.
    """

    # ==================================================================================
    # PARTE 1: CAMINHO MÍNIMO NÃO-PONDERADO & MÉTRICAS TOPOLÓGICAS (BFS)
    # ==================================================================================
    def bfs_caminho_minimo(self, inicio: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
        """
        Executa BFS para calcular distâncias em número de saltos e árvore de predecessores.
        Complexidade: O(V + E)
        """
        if inicio not in self._adj:
            raise KeyError(f"Vértice '{inicio}' não existe no grafo.")

        dist: Dict[str, float] = {v: float("inf") for v in self.vertices}
        pred: Dict[str, Optional[str]] = {v: None for v in self.vertices}

        dist[inicio] = 0.0
        fila: deque[str] = deque([inicio])

        while fila:
            u = fila.popleft()
            for v in self.obter_vizinhos(u):
                if dist[v] == float("inf"):
                    dist[v] = dist[u] + 1.0
                    pred[v] = u
                    fila.append(v)

        return dist, pred

    def rota_bfs(self, origem: str, destino: str) -> Tuple[List[str], int]:
        """Reconstrói a rota com menor número de saltos via BFS."""
        dist, pred = self.bfs_caminho_minimo(origem)
        if dist[destino] == float("inf"):
            return [], -1

        caminho: List[str] = []
        atual: Optional[str] = destino
        while atual is not None:
            caminho.append(atual)
            atual = pred[atual]
        caminho.reverse()
        return caminho, int(dist[destino])

    def metricas_topologicas(self) -> Dict[str, Any]:
        """
        Calcula as métricas topológicas de centralidade baseadas em caminhos mínimos:
        - Excentricidade de cada vértice: epsilon(v) = max_{u} dist(v, u)
        - Raio: rad(G) = min_{v} epsilon(v)
        - Diâmetro: diam(G) = max_{v} epsilon(v)
        - Centro: Z(G) = {v | epsilon(v) == rad(G)}
        """
        if self._direcionado:
            raise ValueError("Métricas de Raio/Diâmetro/Centro exigem grafo não-direcionado conexo.")

        excentricidades: Dict[str, int] = {}
        for v in self.vertices:
            dist, _ = self.bfs_caminho_minimo(v)
            dists_validas = [int(d) for d in dist.values() if d != float("inf")]
            if len(dists_validas) < self.total_vertices:
                # Grafo desconexo
                excentricidades[v] = -1
            else:
                excentricidades[v] = max(dists_validas)

        if any(e == -1 for e in excentricidades.values()):
            raise ValueError("O grafo é desconexo; diâmetro e raio são infinitos.")

        raio = min(excentricidades.values())
        diametro = max(excentricidades.values())
        centro = [v for v, e in excentricidades.items() if e == raio]

        return {
            "excentricidades": excentricidades,
            "raio": raio,
            "diametro": diametro,
            "centro": centro,
        }

    # ==================================================================================
    # PARTE 2: ALGORITMO DE DIJKSTRA (GRAFOS PONDERADOS COM PESOS NÃO-NEGATIVOS)
    # ==================================================================================
    def dijkstra(self, origem: str) -> Tuple[Dict[str, float], Dict[str, Optional[str]], List[str]]:
        """
        Calcula os caminhos mínimos a partir de 'origem' usando o Algoritmo de Dijkstra.
        Utiliza Min-Heap binária (heapq) para extração gulosa do vértice mais próximo.
        Complexidade: O((V + E) log V)
        
        Retorna:
          - dist: dicionário com as distâncias mínimas acumuladas
          - pred: dicionário com os predecessores (árvore de caminhos mínimos)
          - ordem_fechamento: ordem em que os nós tiveram suas distâncias definitivas fixadas
        """
        if origem not in self._adj:
            raise KeyError(f"Vértice de origem '{origem}' não existe no grafo.")

        # Validação essencial: pesos negativos quebram Dijkstra!
        for u, vizinhos in self._adj.items():
            for v, peso in vizinhos.items():
                if peso < 0:
                    raise ValueError(
                        f"Aresta ({u}, {v}) possui peso negativo ({peso}). "
                        "O Algoritmo de Dijkstra exige pesos não-negativos!"
                    )

        dist: Dict[str, float] = {v: float("inf") for v in self.vertices}
        pred: Dict[str, Optional[str]] = {v: None for v in self.vertices}
        visitados: Set[str] = set()
        ordem_fechamento: List[str] = []

        dist[origem] = 0.0
        # Fila de prioridades: tuplas (distancia_atual, vertice)
        heap: List[Tuple[float, str]] = [(0.0, origem)]

        while heap:
            dist_u, u = heapq.heappop(heap)

            # Se o vértice já foi fechado com uma distância menor, ignora duplicata no heap
            if u in visitados:
                continue

            visitados.add(u)
            ordem_fechamento.append(u)

            # Relaxamento de todas as arestas saintes de u
            for v, peso_uv in self._adj[u].items():
                if v not in visitados:
                    nova_dist = dist_u + peso_uv
                    if nova_dist < dist[v]:
                        dist[v] = nova_dist
                        pred[v] = u
                        heapq.heappush(heap, (nova_dist, v))

        return dist, pred, ordem_fechamento

    def caminho_minimo_dijkstra(self, origem: str, destino: str) -> Tuple[List[str], float]:
        """
        Calcula e reconstrói o caminho de custo mínimo entre origem e destino.
        Retorna (lista_de_vertices, custo_total).
        """
        dist, pred, _ = self.dijkstra(origem)
        if dist[destino] == float("inf"):
            return [], float("inf")

        rota: List[str] = []
        atual: Optional[str] = destino
        while atual is not None:
            rota.append(atual)
            atual = pred[atual]
        rota.reverse()
        return rota, dist[destino]


# ======================================================================================
# DEMONSTRAÇÃO DO CONTRAEXEMPLO: POR QUE O BFS FALHA EM GRAFOS PONDERADOS?
# ======================================================================================
def demonstrar_falha_bfs():
    print("\n" + "=" * 80)
    print("CONTRAEXEMPLO DIDÁTICO: POR QUE O BFS FALHA EM GRAFOS PONDERADOS?")
    print("=" * 80)
    print("""
Considere o seguinte grafo rodoviário entre a Origem (A) e o Destino (C):
  - Rota 1 (Direta, 1 salto):  A ──(100 km)──> C
  - Rota 2 (Via B, 2 saltos):   A ──(10 km)───> B ──(10 km)───> C

O que cada algoritmo escolhe?
""")

    g = GrafoCaminhoMinimo(direcionado=True)
    g.adicionar_aresta("A", "C", 100.0)
    g.adicionar_aresta("A", "B", 10.0)
    g.adicionar_aresta("B", "C", 10.0)

    # 1. Execução do BFS (métrica: número de saltos)
    rota_bfs, saltos = g.rota_bfs("A", "C")
    # Calcula a quilometragem real dessa rota do BFS
    km_bfs = sum(g.obter_peso(rota_bfs[i], rota_bfs[i+1]) for i in range(len(rota_bfs)-1))

    # 2. Execução do Dijkstra (métrica: peso acumulado)
    rota_dijkstra, km_dijkstra = g.caminho_minimo_dijkstra("A", "C")

    print(f"Decisão do BFS (conta apenas saltos):")
    print(f"  Rota Escolhida: {' -> '.join(rota_bfs)}")
    print(f"  Saltos: {saltos} salto  |  Distância Real: {km_bfs:.1f} km  [ERRO GRAVE: Rota mais longa!]")

    print(f"\nDecisão do Dijkstra (avalia pesos das rodovias):")
    print(f"  Rota Escolhida: {' -> '.join(rota_dijkstra)}")
    print(f"  Saltos: {len(rota_dijkstra)-1} saltos |  Distância Real: {km_dijkstra:.1f} km  [ÓTIMO GLOBAL]")

    print("\nConclusão: O BFS é míope em relação aos pesos! Ele pressupõe que todas as arestas")
    print("têm custo unitário (w = 1). Para rodovias com distâncias reais, DIJKSTRA é mandatário.")


# ======================================================================================
# ESTUDO DE CASO 1: ROTEAMENTO REGIONAL DE SAFRA (BEBEDOURO - BARRETOS - RIBEIRÃO PRETO)
# ======================================================================================
def executar_caso_1_logistica_safra():
    print("\n" + "#" * 80)
    print("ESTUDO DE CASO 1: ROTEAMENTO DE SAFRA DE CITROS E CANA NO INTERIOR DE SP")
    print("#" * 80)
    print("""
Contexto:
  Uma cooperativa agrícola precisa transportar a colheita de citros de fazendas em
  Monte Azul Paulista e Taquaritinga até a Usina Central de Processamento em Bebedouro
  e o terminal de exportação em Ribeirão Preto.
  As rodovias (SP-326 Brig. Faria Lima, SP-330 Anhanguera, etc.) possuem distâncias em km.
""")

    malha = GrafoCaminhoMinimo(direcionado=False)

    # Rodovias da região (distâncias rodoviárias reais em km)
    rodovias = [
        ("Monte_Azul", "Bebedouro", 32.0),
        ("Monte_Azul", "Severinia", 24.0),
        ("Severinia", "Olimpia", 22.0),
        ("Olimpia", "Barretos", 35.0),
        ("Bebedouro", "Barretos", 48.0),
        ("Bebedouro", "Pitangueiras", 38.0),
        ("Bebedouro", "Taquaritinga", 54.0),
        ("Pitangueiras", "Sertaozinho", 36.0),
        ("Taquaritinga", "Jaboticabal", 35.0),
        ("Jaboticabal", "Sertaozinho", 42.0),
        ("Sertaozinho", "Ribeirao_Preto", 21.0),
        ("Barretos", "Colina", 20.0),
        ("Colina", "Bebedouro", 22.0),
    ]

    for u, v, km in rodovias:
        malha.adicionar_aresta(u, v, km)

    origem = "Monte_Azul"
    destino = "Ribeirao_Preto"

    dist, pred, fechamento = malha.dijkstra(origem)
    rota, total_km = malha.caminho_minimo_dijkstra(origem, destino)

    print(f"Ordem de Fechamento Definitivo dos Vértices pelo Dijkstra a partir de {origem}:")
    print("  " + " -> ".join(fechamento))

    print(f"\nTabela de Distâncias Mínimas a partir de {origem}:")
    print(f"{'Cidade / Ponto Logístico':<25} | {'Distância Mínima (km)':<22} | {'Predecessor na Rota':<20}")
    print("-" * 72)
    for cidade in sorted(malha.vertices):
        pai = pred[cidade] if pred[cidade] else "(Origem)"
        print(f"{cidade:<25} | {dist[cidade]:<22.1f} | {pai:<20}")

    print(f"\nRota de Menor Custo de Frete de {origem} até {destino}:")
    print("  " + " -> ".join(rota))
    print(f"  Quilometragem Total: {total_km:.1f} km")


# ======================================================================================
# ESTUDO DE CASO 2: ESCOLHA DO HUB LOGÍSTICO POR CENTRALIDADE (CENTRO DO GRAFO)
# ======================================================================================
def executar_caso_2_hub_logistico():
    print("\n" + "#" * 80)
    print("ESTUDO DE CASO 2: ONDE CONSTRUIR O NOVO SILO REGIONAL? (CENTRO DO GRAFO)")
    print("#" * 80)
    print("""
Contexto:
  A diretoria de operações da cooperativa precisa definir em qual município instalar
  o novo Centro de Distribuição e Silo Regional de Insumos.
  O critério estratégico de engenharia é MINIMIZAR O TEMPO MÁXIMO DE ATENDIMENTO
  a qualquer cooperativa da rede (Problema do Centro do Grafo / Minimax).
""")

    rede = GrafoCaminhoMinimo(direcionado=False)
    # Rede de 6 unidades regionais
    conexoes = [
        ("Bebedouro", "Pitangueiras"),
        ("Bebedouro", "Colina"),
        ("Bebedouro", "Monte_Azul"),
        ("Colina", "Barretos"),
        ("Pitangueiras", "Sertaozinho"),
        ("Sertaozinho", "Ribeirao_Preto"),
        ("Monte_Azul", "Severinia"),
    ]

    for u, v in conexoes:
        rede.adicionar_aresta(u, v)

    metricas = rede.metricas_topologicas()

    print("Métricas Topológicas de Centralidade da Rede Regional:")
    print(f"{'Município':<20} | {'Excentricidade epsilon(v)':<25}")
    print("-" * 50)
    for mun in sorted(rede.vertices):
        print(f"{mun:<20} | {metricas['excentricidades'][mun]} saltos")

    print(f"\nRaio da Rede (rad(G)):     {metricas['raio']} saltos")
    print(f"Diâmetro da Rede (diam(G)): {metricas['diametro']} saltos (distância máxima entre extremos)")
    print(f"Centro da Rede (Z(G)):     {metricas['centro']}")

    print("\nParecer Técnico de Engenharia de Dados:")
    print(f"  O município de {metricas['centro'][0]} é o CENTRO DO GRAFO, pois possui a menor excentricidade ({metricas['raio']}).")
    print(f"  Instalar o Hub em {metricas['centro'][0]} garante que qualquer outra cidade seja atendida em no máximo {metricas['raio']} saltos!")


# ======================================================================================
# ESTUDO DE CASO 3: ROTEAMENTO COM CUSTO COMPOSTO (DIESEL + PEDÁGIO)
# ======================================================================================
def executar_caso_3_custo_composto():
    print("\n" + "#" * 80)
    print("ESTUDO DE CASO 3: ROTEAMENTO MULTICRITÉRIO COM DIESEL E PEDÁGIO")
    print("#" * 80)
    print("""
Contexto:
  Caminhões bitrem carregados com 40 toneladas de açúcar possuem custo operacional
  de R$ 4,50 por km rodado (combustível, pneus, manutenção) + tarifas fixas de pedágio.
  Frequentemente, a rota mais curta em km NÃO é a mais barata em reais devido aos pedágios!
""")

    malha_financeira = GrafoCaminhoMinimo(direcionado=False)

    # Definindo trechos com: Distância (km) e Pedágio (R$)
    # Custo da aresta = (km * 4.50) + pedagio
    trechos = [
        ("Usina_Central", "Ponto_A", 40.0, 35.0),   # 40km, R$35 pedágio -> R$215,00
        ("Usina_Central", "Ponto_B", 60.0, 0.0),    # 60km, sem pedágio  -> R$270,00
        ("Ponto_A", "Terminal_Ferroviario", 50.0, 45.0), # 50km, R$45 pedágio -> R$270,00
        ("Ponto_B", "Terminal_Ferroviario", 35.0, 10.0), # 35km, R$10 pedágio -> R$167,50
    ]

    for u, v, km, ped in trechos:
        custo_total = (km * 4.50) + ped
        malha_financeira.adicionar_aresta(u, v, custo_total)

    rota, custo = malha_financeira.caminho_minimo_dijkstra("Usina_Central", "Terminal_Ferroviario")

    print("Comparativo de Custos Operacionais por Trecho:")
    for u, v, km, ped in trechos:
        c = (km * 4.50) + ped
        print(f"  {u} <-> {v}: {km:.0f} km | Pedágio: R$ {ped:.2f} | Custo Total: R$ {c:.2f}")

    print(f"\nRota de Mínimo Custo Financeiro:")
    print("  " + " -> ".join(rota))
    print(f"  Custo Total da Viagem: R$ {custo:.2f}")


# ======================================================================================
# EXERCÍCIOS DE SALA RESOLVIDOS
# ======================================================================================
def executar_exercicio_1_metricas():
    print("\n" + "=" * 80)
    print("EXERCÍCIO 1 (SLIDES): CÁLCULO DE EXCENTRICIDADE, RAIO, DIÂMETRO E CENTRO")
    print("=" * 80)
    print("Grafo: A-B, B-C, C-D, D-E (Linha de 5 nós: P5) com ramo adicional C-F")

    g = GrafoCaminhoMinimo(direcionado=False)
    g.adicionar_aresta("A", "B")
    g.adicionar_aresta("B", "C")
    g.adicionar_aresta("C", "D")
    g.adicionar_aresta("D", "E")
    g.adicionar_aresta("C", "F")

    met = g.metricas_topologicas()
    print("\nResultados Oficiais do Gabarito:")
    for v in sorted(g.vertices):
        print(f"  Excentricidade de {v}: {met['excentricidades'][v]}")
    print(f"Raio do Grafo:     {met['raio']}")
    print(f"Diâmetro do Grafo: {met['diametro']}")
    print(f"Centro do Grafo:   {met['centro']}")


def executar_exercicio_2_dijkstra():
    print("\n" + "=" * 80)
    print("EXERCÍCIO 2 (SLIDES): EXECUÇÃO COMPLETA DE DIJKSTRA COM TABELA DE ITERAÇÕES")
    print("=" * 80)
    print("Grafo com 6 vértices: S, A, B, C, D, E")

    g = GrafoCaminhoMinimo(direcionado=True)
    g.adicionar_aresta("S", "A", 4.0)
    g.adicionar_aresta("S", "B", 2.0)
    g.adicionar_aresta("B", "A", 1.0)
    g.adicionar_aresta("B", "C", 5.0)
    g.adicionar_aresta("A", "C", 2.0)
    g.adicionar_aresta("A", "D", 3.0)
    g.adicionar_aresta("C", "D", 4.0)
    g.adicionar_aresta("C", "E", 5.0)
    g.adicionar_aresta("D", "E", 1.0)

    dist, pred, fechamento = g.dijkstra("S")

    print("\nOrdem de Fechamento Definitivo pelo Min-Heap:")
    print("  " + " -> ".join(fechamento))

    print("\nTabela Final de Distâncias e Predecessores (Gabarito Oficial):")
    print(f"{'Vértice':<10} | {'Distância d[v]':<16} | {'Predecessor pi[v]':<18}")
    print("-" * 48)
    for v in sorted(g.vertices):
        pai = pred[v] if pred[v] else "-"
        print(f"{v:<10} | {dist[v]:<16.1f} | {pai:<18}")

    rota_s_e, custo_s_e = g.caminho_minimo_dijkstra("S", "E")
    print(f"\nCaminho Mínimo de S até E: {' -> '.join(rota_s_e)} (Custo: {custo_s_e:.1f})")


# ======================================================================================
# BATERIA DE TESTES AUTOMATIZADOS DE INTEGRIDADE
# ======================================================================================
def executar_testes_unitarios():
    g = GrafoCaminhoMinimo(direcionado=False)
    g.adicionar_aresta("A", "B", 4.0)
    g.adicionar_aresta("A", "C", 2.0)
    g.adicionar_aresta("C", "B", 1.0)
    g.adicionar_aresta("B", "D", 5.0)
    g.adicionar_aresta("C", "D", 8.0)

    # Teste Dijkstra
    dist, pred, _ = g.dijkstra("A")
    assert dist["A"] == 0.0
    assert dist["C"] == 2.0
    assert dist["B"] == 3.0  # A -> C -> B (2 + 1 = 3 < 4)
    assert dist["D"] == 8.0  # A -> C -> B -> D (2 + 1 + 5 = 8)
    assert pred["B"] == "C"

    # Teste Rota
    rota, c = g.caminho_minimo_dijkstra("A", "D")
    assert rota == ["A", "C", "B", "D"]
    assert c == 8.0

    # Teste de Validação de Peso Negativo
    g_neg = GrafoCaminhoMinimo(direcionado=True)
    g_neg.adicionar_aresta("X", "Y", -5.0)
    try:
        g_neg.dijkstra("X")
        assert False, "Deveria ter lançado ValueError para peso negativo"
    except ValueError:
        pass

    # Teste Métricas Topológicas
    g_linha = GrafoCaminhoMinimo(direcionado=False)
    g_linha.adicionar_aresta("1", "2")
    g_linha.adicionar_aresta("2", "3")
    g_linha.adicionar_aresta("3", "4")
    met = g_linha.metricas_topologicas()
    assert met["diametro"] == 3
    assert met["raio"] == 2
    assert set(met["centro"]) == {"2", "3"}

    print(">>> TODOS OS TESTES UNITÁRIOS DE INTEGRIDADE PASSARAM COM SUCESSO! <<<")


# ======================================================================================
# PROGRAMA PRINCIPAL
# ======================================================================================
if __name__ == "__main__":
    executar_testes_unitarios()
    demonstrar_falha_bfs()
    executar_caso_1_logistica_safra()
    executar_caso_2_hub_logistico()
    executar_caso_3_custo_composto()
    executar_exercicio_1_metricas()
    executar_exercicio_2_dijkstra()
