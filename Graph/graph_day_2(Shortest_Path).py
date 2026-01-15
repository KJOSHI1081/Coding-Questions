"""
TOPIC: GRAPHS - DAY 2
Shortest Path Algorithms and Related Patterns
    1. SHORTEST PATH IN UNWEIGHTED GRAPH (BFS)
    2. DIJKSTRA'S ALGORITHM (Shortest Path in Weighted Graph)
    3. BELLMAN-FORD ALGORITHM (Handles Negative Weights)
    4. FLOYD-WARSHALL ALGORITHM (All-Pairs Shortest Path)
    5. NUMBER OF ISLANDS (Connected Components)

5 Shortest Path Interview Questions with Test Cases
"""

import heapq
from collections import deque, defaultdict

class ShortestPathSolutions:
    """
    STAFF LEVEL INSIGHT:
    The choice of shortest path algorithm depends on:
    1. Weighted vs Unweighted edges
    2. Presence of negative weights
    3. Single-source vs all-pairs requirement
    4. Graph density (sparse vs dense)
    
    MEMORY CONSIDERATION:
    Dijkstra is preferred for sparse graphs with non-negative weights.
    Floyd-Warshall is only feasible for small, dense graphs (V <= 500).
    """

    # =========================================================================
    # 1. SHORTEST PATH IN UNWEIGHTED GRAPH (BFS)
    # =========================================================================
    def shortest_path_unweighted(self, n: int, edges: list, start: int, end: int) -> int:
        """
        Find shortest path in unweighted graph using BFS.
        
        EXPLANATION:
        In unweighted graphs, all edges have equal weight (usually 1).
        BFS naturally finds the shortest path because it explores level-by-level.
        
        ALGORITHM:
        1. Start from the source node
        2. Use a queue to explore neighbors layer by layer
        3. Track distance from start for each node
        4. Return distance when we reach the end node
        
        TIME: O(V + E)
        SPACE: O(V)
        
        EXAMPLE:
        Graph: 0-1-2-3, find shortest path from 0 to 3
        BFS explores: 0 (dist=0) -> 1 (dist=1) -> 2 (dist=2) -> 3 (dist=3)
        Result: 3
        
        WHY STAFF LEVEL:
        - Foundation for understanding Dijkstra
        - Often faster than Dijkstra when weights are uniform
        - Demonstrates the power of choosing the right traversal algorithm
        """
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        if start == end:
            return 0
        
        visited = set([start])
        queue = deque([(start, 0)])  # (node, distance)
        
        while queue:
            node, dist = queue.popleft()
            
            for neighbor in graph[node]:
                if neighbor == end:
                    return dist + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return -1  # No path found

    # =========================================================================
    # 2. DIJKSTRA'S ALGORITHM
    # =========================================================================
    def dijkstra(self, n: int, edges: list, start: int) -> list:
        """
        Find shortest path from start to all other nodes using Dijkstra's Algorithm.
        
        EXPLANATION:
        Dijkstra's algorithm finds the shortest path in a weighted graph with
        NON-NEGATIVE weights.
        
        CORE IDEA (Greedy):
        1. Maintain distances from start to all nodes (initially infinity)
        2. Use a min-heap to always process the closest unvisited node next
        3. For each processed node, relax its neighbors:
           If (distance[current] + edge_weight) < distance[neighbor],
           update distance[neighbor]
        4. Continue until all nodes are processed or heap is empty
        
        WHY IT WORKS (PROOF):
        Once we process a node with a certain distance, we've found the
        SHORTEST path to that node. Why? Because:
        - We always process the closest node first (min-heap)
        - All edge weights are non-negative
        - Any other path would be longer (since we'd go through heavier edges)
        
        TIME: O((V + E) log V) with min-heap
        SPACE: O(V)
        
        EXAMPLE:
        Graph: 0--(1)--1--(2)--2
               |          |
               (4)       (1)
               |          |
               3----------+
        
        Dijkstra from 0:
        - distances = [0, 1, 3, 4]
        - Process 0: update 1 and 3
        - Process 1: update 2 (cost becomes min(inf, 1+2) = 3)
        - Process 2: update 3 (cost becomes min(4, 3+1) = 4, no change)
        - Process 3: done
        
        WHY STAFF LEVEL:
        - Most used shortest path algorithm in real systems (GPS, routing)
        - Understanding heap optimization
        - Can be extended to K shortest paths
        - Foundation for understanding A* algorithm
        """
        # Build adjacency list with weights
        graph = defaultdict(list)
        for u, v, weight in edges:
            graph[u].append((v, weight))
            graph[v].append((u, weight))
        
        # Initialize distances
        distances = [float('inf')] * n
        distances[start] = 0
        
        # Min-heap: (distance, node)
        min_heap = [(0, start)]
        visited = set()
        
        while min_heap:
            curr_dist, node = heapq.heappop(min_heap)
            
            if node in visited:
                continue
            
            visited.add(node)
            
            # Relax edges
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    new_dist = curr_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(min_heap, (new_dist, neighbor))
        
        return distances

    # =========================================================================
    # 3. BELLMAN-FORD ALGORITHM
    # =========================================================================
    def bellman_ford(self, n: int, edges: list, start: int) -> list:
        """
        Find shortest path allowing negative weights (but no negative cycles).
        
        EXPLANATION:
        Bellman-Ford is slower than Dijkstra but handles negative edge weights.
        It cannot handle negative-weight cycles (which would make shortest path
        undefined, as we could keep looping to reduce the path cost).
        
        ALGORITHM:
        1. Initialize distances (start=0, others=infinity)
        2. Repeat V-1 times:
           - For each edge (u, v, weight):
             If distance[u] + weight < distance[v],
             update distance[v]
        3. Check for negative cycles (optional):
           - Run one more iteration; if anything changes, negative cycle exists
        
        WHY V-1 ITERATIONS?
        The shortest path in a graph with V vertices can have at most V-1 edges.
        After V-1 iterations, all shortest paths are guaranteed to be found.
        
        TIME: O(V * E)
        SPACE: O(V)
        
        TRADE-OFF vs DIJKSTRA:
        - Dijkstra: O((V+E) log V), but NO negative weights
        - Bellman-Ford: O(V*E), handles negative weights
        
        WHY STAFF LEVEL:
        - Understanding why algorithm needs V-1 iterations
        - Detecting negative cycles (financial arbitrage detection)
        - Proof that shortest path has at most V-1 edges
        """
        distances = [float('inf')] * n
        distances[start] = 0
        
        # Relax edges V-1 times
        for _ in range(n - 1):
            for u, v, weight in edges:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
        
        # Check for negative cycles (optional)
        for u, v, weight in edges:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                print("WARNING: Negative cycle detected!")
                return None
        
        return distances

    # =========================================================================
    # 4. FLOYD-WARSHALL ALGORITHM
    # =========================================================================
    def floyd_warshall(self, n: int, edges: list) -> list:
        """
        Find shortest path between ALL pairs of vertices.
        
        EXPLANATION:
        Floyd-Warshall uses dynamic programming to compute shortest paths
        between all pairs of vertices.
        
        KEY IDEA (Dynamic Programming):
        dist[i][j] = shortest path from i to j
        
        For each intermediate vertex k:
            For each pair (i, j):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        This means: "Can we get a shorter path from i to j by going through k?"
        
        ALGORITHM:
        1. Initialize dist matrix:
           - dist[i][i] = 0 (distance to self is 0)
           - dist[i][j] = weight if edge exists, else infinity
        2. For k = 0 to n-1:
           For i = 0 to n-1:
               For j = 0 to n-1:
                   dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        TIME: O(V^3)
        SPACE: O(V^2)
        
        WHEN TO USE:
        - Small graphs (V <= 500)
        - Need all-pairs shortest path
        - Can handle negative weights (no negative cycles)
        - Single-source queries are rare (use Dijkstra instead)
        
        WHY STAFF LEVEL:
        - Demonstrates nested DP structure
        - Understanding why intermediate vertex matters
        - Recognizing when O(V^3) is acceptable
        - Transitive closure computation
        """
        # Initialize distance matrix
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        
        for u, v, weight in edges:
            dist[u][v] = min(dist[u][v], weight)  # Handle multiple edges
            dist[v][u] = min(dist[v][u], weight)  # For undirected
        
        # Floyd-Warshall DP
        for k in range(n):  # Intermediate vertex
            for i in range(n):
                for j in range(n):
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        
        return dist

    # =========================================================================
    # 5. NUMBER OF ISLANDS (Connected Components)
    # =========================================================================
    def num_islands(self, grid: list) -> int:
        """
        Count the number of islands in a 2D grid.
        
        EXPLANATION:
        An island is a group of connected 1s (horizontally or vertically adjacent).
        Water is represented by 0s.
        
        PROBLEM:
        Given a 2D grid of '0's and '1's, count distinct islands.
        
        Example:
        Grid:
        1 1 0 0 0
        1 0 0 1 0
        0 0 1 0 1
        0 0 0 0 0
        
        Islands: 3 (one at top-left, one at middle-right, one at bottom-right)
        
        ALGORITHM:
        1. Iterate through each cell in the grid
        2. When we find a '1' that hasn't been visited:
           - Increment island count
           - Use DFS to mark all connected '1's as visited
        3. Continue until entire grid is processed
        
        WHY DFS? (Not BFS)
        - We only care about counting islands, not finding shortest path
        - DFS is simpler for 2D grid traversal and avoids queue overhead
        - Both have same time complexity
        
        TIME: O(rows * cols)
        SPACE: O(rows * cols) - for visited set and recursion stack
        
        OPTIMIZATION:
        - Modify grid in-place instead of using separate visited set
        - This reduces space to O(1) extra space (excluding recursion stack)
        
        WHY STAFF LEVEL:
        - Demonstrates connected components concept
        - 2D grid traversal patterns
        - Can be extended to count island perimeters, largest island, etc.
        """
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        visited = set()
        island_count = 0
        
        def dfs(r, c):
            # Mark as visited
            if (r, c) in visited:
                return
            visited.add((r, c))
            
            # Explore 4 adjacent cells (up, down, left, right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] == '1' and (nr, nc) not in visited:
                        dfs(nr, nc)
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1' and (r, c) not in visited:
                    dfs(r, c)
                    island_count += 1
        
        return island_count


# =========================================================================
# MAIN EXECUTION BLOCK
# =========================================================================

if __name__ == "__main__":
    sol = ShortestPathSolutions()
    
    print("=" * 70)
    print("GRAPH DAY 2: SHORTEST PATH ALGORITHMS")
    print("=" * 70)
    
    print("\n--- 1. Shortest Path in Unweighted Graph (BFS) ---")
    edges_unweighted = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 3)]
    result = sol.shortest_path_unweighted(5, edges_unweighted, 0, 3)
    print(f"Shortest path from 0 to 3: {result} (Expected: 2)")
    
    print("\n--- 2. Dijkstra's Algorithm ---")
    edges_weighted = [(0, 1, 1), (1, 2, 2), (2, 3, 1), (0, 4, 4), (4, 3, 1)]
    distances = sol.dijkstra(5, edges_weighted, 0)
    print(f"Shortest distances from 0: {distances}")
    print(f"Expected: [0, 1, 3, 2, 4]")
    
    print("\n--- 3. Bellman-Ford Algorithm ---")
    distances_bf = sol.bellman_ford(5, edges_weighted, 0)
    print(f"Shortest distances from 0: {distances_bf}")
    print(f"Expected: [0, 1, 3, 2, 4]")
    
    print("\n--- 4. Floyd-Warshall Algorithm ---")
    all_pairs = sol.floyd_warshall(5, edges_weighted)
    print(f"Shortest distances between all pairs:")
    for i, row in enumerate(all_pairs):
        print(f"  From {i}: {row}")
    
    print("\n--- 5. Number of Islands ---")
    grid = [
        ['1', '1', '0', '0', '0'],
        ['1', '0', '0', '1', '0'],
        ['0', '0', '1', '0', '1'],
        ['0', '0', '0', '0', '0']
    ]
    islands = sol.num_islands(grid)
    print(f"Number of islands: {islands} (Expected: 3)")
    
    print("\n" + "=" * 70)
