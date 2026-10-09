"""
Algoritmos Avancados - Aula 16: Introducao a Teoria dos Grafos
Prof. Me. Douglas Samuel Goncalves
Fatec Bebedouro - Jorge Caram Sabbag
Tecnologia em Big Data no Agronegocio
"""

from typing import Optional


class Grafo:
    """
    Representacao de Grafo utilizando Dicionario de Adjacencia.
    Suporta grafos direcionados, nao-direcionados e ponderados.
    """

    def __init__(self, direcionado: bool = False):
        self._direcionado: bool = direcionado
        # Estrutura: {vertice_origem: {vertice_destino: peso}}
        self._adj: dict[str, dict[str, float]] = {}

    @property
    def eh_direcionado(self) -> bool:
        """Indica se o grafo e direcionado."""
        return self._direcionado

    @property
    def total_vertices(self) -> int:
        """Retorna a cardinalidade do conjunto de vertices (|V|)."""
        return len(self._adj)

    def adicionar_vertice(self, vertice: str) -> bool:
        """Adiciona um novo vertice isolado caso nao exista."""
        if vertice not in self._adj:
            self._adj[vertice] = {}
            return True
        return False

    def adicionar_aresta(self, u: str, v: str, peso: float = 1.0) -> None:
        """Cria uma aresta ponderada entre u e v."""
        # Garante a existencia dos vertices
        self.adicionar_vertice(u)
        self.adicionar_vertice(v)

        # Insere a aresta de u para v
        self._adj[u][v] = peso

        # Se for nao-direcionado, insere a aresta simetrica de v para u
        if not self._direcionado:
            self._adj[v][u] = peso

    def remover_aresta(self, u: str, v: str) -> bool:
        """Remove a aresta entre u e v se ela existir. Retorna True se removida."""
        if u in self._adj and v in self._adj[u]:
            del self._adj[u][v]
            if not self._direcionado and v in self._adj and u in self._adj[v]:
                del self._adj[v][u]
            return True
        return False

    def obter_vizinhos(self, vertice: str) -> list[str]:
        """Retorna a lista de vizinhos adjacentes ao vertice."""
        if vertice in self._adj:
            return list(self._adj[vertice].keys())
        return []

    def obter_peso(self, u: str, v: str) -> Optional[float]:
        """Retorna o peso da aresta (u, v) ou None se inexistente."""
        return self._adj.get(u, {}).get(v, None)

    def grau(self, vertice: str) -> int:
        """Retorna o grau de saida ou grau total do vertice."""
        if vertice not in self._adj:
            raise KeyError(f"Vertice '{vertice}' nao encontrado no grafo.")
        return len(self._adj[vertice])

    def vertice_maior_grau(self) -> str:
        """Retorna o vertice de maior grau no grafo (grau maximo Delta(G))."""
        if not self._adj:
            raise ValueError("O grafo esta vazio.")
        return max(self._adj.keys(), key=lambda v: len(self._adj[v]))

    def tem_aresta(self, u: str, v: str) -> bool:
        """Verifica a existencia de aresta direta entre u e v em tempo O(1)."""
        return (u in self._adj) and (v in self._adj[u])

    def total_arestas(self) -> int:
        """Calcula o numero total de arestas no grafo."""
        total = sum(len(vizinhos) for vizinhos in self._adj.values())
        return total if self._direcionado else total // 2

    def densidade(self) -> float:
        """
        Calcula a densidade do grafo. Retorna float no intervalo [0.0, 1.0].
        """
        n = self.total_vertices
        if n <= 1:
            return 0.0

        m = self.total_arestas()
        if self._direcionado:
            max_arestas = n * (n - 1)
        else:
            max_arestas = (n * (n - 1)) / 2

        return m / max_arestas

    def __str__(self) -> str:
        linhas = [f"=== Grafo ({'Direcionado' if self._direcionado else 'Nao-Direcionado'}) ==="]
        for v, vizinhos in self._adj.items():
            conexoes = ", ".join(f"{dest}(w={peso})" for dest, peso in vizinhos.items())
            linhas.append(f"  [{v}] -> {conexoes if conexoes else 'Nenhum vizinho'}")
        return "\n".join(linhas)


# ====================================================================
# Execucao do Estudo de Caso Pratico (Slides 35-37)
# ====================================================================
if __name__ == "__main__":
    malha_agro = Grafo(direcionado=False)

    # Adicionando trechos rodoviarios e distancias em km
    malha_agro.adicionar_aresta("Talhao_1", "Silo_Norte", 12.0)
    malha_agro.adicionar_aresta("Talhao_2", "Silo_Norte", 8.0)
    malha_agro.adicionar_aresta("Talhao_2", "Silo_Sul", 15.0)
    malha_agro.adicionar_aresta("Talhao_3", "Silo_Sul", 10.0)
    malha_agro.adicionar_aresta("Silo_Norte", "Usina_Central", 28.0)
    malha_agro.adicionar_aresta("Silo_Sul", "Usina_Central", 22.0)

    print(malha_agro)
    print(f"Total de Vertices: {malha_agro.total_vertices}")
    print(f"Total de Arestas:  {malha_agro.total_arestas()}")
    print(f"Vizinhos do Silo Norte: {malha_agro.obter_vizinhos('Silo_Norte')}")
    print(f"Grau da Usina Central: {malha_agro.grau('Usina_Central')}")
    print(f"Densidade da Malha: {malha_agro.densidade():.4f}")
    print(f"Vertice de Maior Grau: {malha_agro.vertice_maior_grau()} (Grau {malha_agro.grau(malha_agro.vertice_maior_grau())})")
