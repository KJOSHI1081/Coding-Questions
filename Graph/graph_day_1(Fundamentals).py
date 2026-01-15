"""
TOPIC: GRAPHS - DAY 1 (FUNDAMENTALS)
Core Graph Concepts and Basic Traversals
    1. GRAPH REPRESENTATION (Adjacency List vs Matrix)
    2. DEPTH-FIRST SEARCH (DFS)
    3. BREADTH-FIRST SEARCH (BFS)
    4. DETECT CYCLE IN UNDIRECTED GRAPH
    5. DETECT CYCLE IN DIRECTED GRAPH (Using Colors)

5 Fundamental Graph Interview Questions with Test Cases
"""

from collections import deque, defaultdict

class GraphSolutions:
    """
    STAFF LEVEL INSIGHT:
    Graph problems often come down to choosing the right traversal (DFS vs BFS)
    and understanding the difference between directed and undirected graphs.
    
    Memory Trade-off: Adjacency List uses O(V + E) space, while Adjacency Matrix 
    uses O(V^2). For sparse graphs (E << V^2), use Adjacency List.
    """

    # =========================================================================
    # 1. GRAPH REPRESENTATION
    # =========================================================================
    def build_adjacency_list(self, n: int, edges: list) -> dict:
        """
        Build an undirected graph using Adjacency List.
        
        EXPLANATION:
        An Adjacency List stores the graph as a dictionary where each key is a 
        vertex and the value is a list of neighbors.
        
        Example: edges = [(0,1), (1,2), (2,0)]
        Result: {0: [1, 2], 1: [0, 2], 2: [1, 0]}
        
        WHY STAFF LEVEL:
        - Memory efficient for sparse graphs
        - Easy to iterate over neighbors
        - Natural fit for DFS and BFS
        """
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)  # For undirected graphs
        return graph

    def build_adjacency_matrix(self, n: int, edges: list) -> list:
        """
        Build an undirected graph using Adjacency Matrix.
        
        EXPLANATION:
        An Adjacency Matrix is a 2D array where matrix[i][j] = 1 if there's 
        an edge between vertices i and j, else 0.
        
        Example: n=3, edges = [(0,1), (1,2)]
        Result: [[0, 1, 0],
                 [1, 0, 1],
                 [0, 1, 0]]
        
        TRADE-OFF:
        - Faster edge lookup: O(1) vs O(degree) for adjacency list
        - But uses O(V^2) space, which is wasteful for sparse graphs
        """
        matrix = [[0] * n for _ in range(n)]
        for u, v in edges:
            matrix[u][v] = 1
            matrix[v][u] = 1  # For undirected graphs
        return matrix

    # =========================================================================
    # 2. DEPTH-FIRST SEARCH (DFS)
    # =========================================================================
    def dfs_recursive(self, graph: dict, start: int, visited=None) -> list:
        """
        Recursive DFS Traversal.
        
        EXPLANATION:
        DFS uses a Stack (implicit via recursion) to explore as far as possible
        along each branch before backtracking.
        
        Order: Start -> Go Deep -> Backtrack
        
        TIME: O(V + E) - Visit each vertex and edge once
        SPACE: O(V) - Recursion stack in worst case (linear graph)
        
        WHY STAFF LEVEL:
        - Understanding when to use recursion vs iteration
        - Recognizing DFS patterns: topological sort, cycle detection, backtracking
        """
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        for neighbor in graph.get(start, []):
            if neighbor not in visited:
                result.extend(self.dfs_recursive(graph, neighbor, visited))
        
        return result

    def dfs_iterative(self, graph: dict, start: int) -> list:
        """
        Iterative DFS using explicit Stack.
        
        WHY STAFF LEVEL:
        - Avoids recursion depth limits for very large graphs
        - More control over the traversal order
        - Easier to understand the stack behavior
        """
        visited = set()
        stack = [start]
        result = []
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)
                # Add neighbors in reverse order to maintain left-to-right traversal
                for neighbor in reversed(graph.get(node, [])):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result

    # =========================================================================
    # 3. BREADTH-FIRST SEARCH (BFS)
    # =========================================================================
    def bfs(self, graph: dict, start: int) -> list:
        """
        BFS Traversal using Queue.
        
        EXPLANATION:
        BFS explores vertices level-by-level. All neighbors are visited before
        going deeper. Uses a Queue (FIFO).
        
        Order: Visit neighbors -> Then neighbors of neighbors
        
        TIME: O(V + E)
        SPACE: O(V) - Queue can hold up to V vertices
        
        WHY STAFF LEVEL:
        - Finds shortest path in unweighted graphs
        - Essential for level-order traversal problems
        - Better cache locality than DFS in some scenarios
        """
        visited = set([start])
        queue = deque([start])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return result

    # =========================================================================
    # 4. DETECT CYCLE IN UNDIRECTED GRAPH
    # =========================================================================
    def has_cycle_undirected(self, n: int, edges: list) -> bool:
        """
        Detect cycle in an undirected graph using DFS.
        
        EXPLANATION:
        A cycle exists if we visit a vertex that's already been visited AND
        it's not the parent of the current vertex.
        
        Example: 0-1-2-0 forms a cycle
        
        ALGORITHM:
        1. For each unvisited vertex, start DFS
        2. During DFS, if we encounter a visited vertex that's not the parent,
           a cycle is found
        3. The parent check is crucial: in undirected graphs, the edge back to
           the parent is not a cycle
        
        TIME: O(V + E)
        SPACE: O(V)
        
        STAFF INSIGHT:
        The parent parameter prevents false positives from the bi-directional
        edge in undirected graphs.
        """
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        visited = set()
        
        def dfs(node, parent):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        return True
                elif neighbor != parent:  # Cycle detected
                    return True
            return False
        
        for i in range(n):
            if i not in visited:
                if dfs(i, -1):
                    return True
        
        return False

    # =========================================================================
    # 5. DETECT CYCLE IN DIRECTED GRAPH (Using Colors)
    # =========================================================================
    def has_cycle_directed(self, n: int, edges: list) -> bool:
        """
        Detect cycle in a directed graph using the Color Method.
        
        EXPLANATION:
        Color-based cycle detection uses three states:
        - WHITE (0): Not visited
        - GRAY (1): Currently being processed (in the recursion stack)
        - BLACK (2): Completely processed
        
        CYCLE DETECTION RULE:
        If we encounter a GRAY node during DFS, a cycle exists because:
        - The GRAY node is an ancestor in the current DFS path
        - We found a back edge (edge pointing to an ancestor)
        
        Example: 1->2->3->1 is a cycle
        When processing 1, we mark it GRAY.
        We visit 2 (mark GRAY), then 3 (mark GRAY).
        From 3, we try to visit 1, which is GRAY -> CYCLE DETECTED!
        
        TIME: O(V + E)
        SPACE: O(V)
        
        STAFF INSIGHT:
        This is more robust than the undirected approach because:
        - Handles self-loops (node pointing to itself)
        - Works for directed acyclic graphs (DAGs) verification
        - Forms the basis for topological sorting
        """
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # 0: WHITE (unvisited), 1: GRAY (in progress), 2: BLACK (finished)
        color = [0] * n
        
        def dfs(node):
            color[node] = 1  # Mark as GRAY (in current path)
            
            for neighbor in graph[node]:
                if color[neighbor] == 1:  # Back edge detected
                    return True
                if color[neighbor] == 0 and dfs(neighbor):  # Explore unvisited
                    return True
            
            color[node] = 2  # Mark as BLACK (finished)
            return False
        
        for i in range(n):
            if color[i] == 0:
                if dfs(i):
                    return True
        
        return False


# =========================================================================
# HELPER FUNCTIONS FOR TESTING
# =========================================================================

def print_graph(graph):
    """Pretty print an adjacency list graph."""
    for node, neighbors in sorted(graph.items()):
        print(f"  {node}: {neighbors}")


# =========================================================================
# MAIN EXECUTION BLOCK
# =========================================================================

if __name__ == "__main__":
    sol = GraphSolutions()
    
    print("=" * 70)
    print("GRAPH DAY 1: FUNDAMENTALS")
    print("=" * 70)
    
    print("\n--- 1. Graph Representation (Adjacency List) ---")
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    adj_list = sol.build_adjacency_list(4, edges)
    print("Edges:", edges)
    print("Adjacency List:")
    print_graph(adj_list)
    
    print("\n--- 2. Graph Representation (Adjacency Matrix) ---")
    matrix = sol.build_adjacency_matrix(4, edges)
    print("Adjacency Matrix:")
    for row in matrix:
        print(" ", row)
    
    print("\n--- 3. DFS Recursive ---")
    dfs_result = sol.dfs_recursive(adj_list, 0)
    print(f"DFS from node 0: {dfs_result}")
    
    print("\n--- 4. DFS Iterative ---")
    dfs_iter_result = sol.dfs_iterative(adj_list, 0)
    print(f"DFS (Iterative) from node 0: {dfs_iter_result}")
    
    print("\n--- 5. BFS ---")
    bfs_result = sol.bfs(adj_list, 0)
    print(f"BFS from node 0: {bfs_result}")
    
    print("\n--- 6. Detect Cycle in Undirected Graph ---")
    cycle_edges_1 = [(0, 1), (1, 2), (2, 0)]  # Has cycle: 0-1-2-0
    cycle_edges_2 = [(0, 1), (1, 2), (2, 3)]  # No cycle
    print(f"Edges {cycle_edges_1} has cycle: {sol.has_cycle_undirected(4, cycle_edges_1)}")
    print(f"Edges {cycle_edges_2} has cycle: {sol.has_cycle_undirected(4, cycle_edges_2)}")
    
    print("\n--- 7. Detect Cycle in Directed Graph ---")
    directed_cycle_1 = [(0, 1), (1, 2), (2, 0)]  # Has cycle: 0->1->2->0
    directed_cycle_2 = [(0, 1), (1, 2), (2, 3)]  # No cycle
    print(f"Edges {directed_cycle_1} has cycle: {sol.has_cycle_directed(4, directed_cycle_1)}")
    print(f"Edges {directed_cycle_2} has cycle: {sol.has_cycle_directed(4, directed_cycle_2)}")
    
    print("\n" + "=" * 70)
