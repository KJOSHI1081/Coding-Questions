"""
TOPIC: GRAPHS - DAY 3 (ADVANCED PATTERNS)
Topological Sort, Strongly Connected Components, and Graph Coloring
    1. TOPOLOGICAL SORT (DFS & Kahn's Algorithm)
    2. STRONGLY CONNECTED COMPONENTS (Kosaraju & Tarjan)
    3. BIPARTITE GRAPH DETECTION
    4. GRAPH COLORING (K-Coloring Problem)
    5. ALIEN DICTIONARY (Topological Sort Application)

5 Advanced Graph Interview Questions with Test Cases
"""

from collections import deque, defaultdict

class AdvancedGraphSolutions:
    """
    STAFF LEVEL INSIGHT:
    Advanced graph problems often involve:
    1. Detecting structural properties (connectivity, planarity)
    2. Ordering constraints (topological sort)
    3. Community detection (strongly connected components)
    4. Coloring/Partitioning problems
    
    These patterns appear in:
    - Build systems (dependency resolution)
    - Social networks (community detection)
    - Register allocation (graph coloring)
    - Natural language processing (alien dictionary)
    """

    # =========================================================================
    # 1. TOPOLOGICAL SORT
    # =========================================================================
    def topological_sort_dfs(self, n: int, edges: list) -> list:
        """
        Find a linear ordering of vertices such that for every edge u->v,
        u comes before v.
        
        EXPLANATION:
        Topological sort is only valid for Directed Acyclic Graphs (DAGs).
        It represents a valid ordering respecting all precedence constraints.
        
        DFS APPROACH:
        1. Perform DFS on the graph
        2. When we finish processing a node (backtrack), add it to result
        3. Reverse the result to get topological order
        
        WHY IT WORKS:
        - When we mark a node as "finished," all its descendants are processed
        - A node can only appear in the ordering after all nodes it depends on
        - Reversing gives us the correct order
        
        TIME: O(V + E)
        SPACE: O(V)
        
        REAL-WORLD EXAMPLE:
        - Course prerequisites: CS101 -> CS102 -> CS201
        - Build system: compile.cpp depends on headers.h
        - Package manager: installing library X requires library Y first
        
        WHY STAFF LEVEL:
        - Understanding when topological sort is applicable
        - Recognizing it's only for DAGs
        - Comparison with Kahn's algorithm
        """
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        visited = set()
        result = []
        
        def dfs(node):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor)
            result.append(node)  # Add after processing all descendants
        
        for i in range(n):
            if i not in visited:
                dfs(i)
        
        return result[::-1]  # Reverse to get topological order

    def topological_sort_kahn(self, n: int, edges: list) -> list:
        """
        Kahn's Algorithm for Topological Sort (using In-degree).
        
        EXPLANATION:
        Kahn's algorithm is a BFS-based approach using in-degrees (incoming edges).
        
        ALGORITHM:
        1. Calculate in-degree for each vertex
        2. Add all vertices with in-degree 0 to a queue (no dependencies)
        3. While queue is not empty:
           - Remove a vertex with in-degree 0
           - Add it to result
           - For each neighbor, decrease its in-degree
           - If neighbor's in-degree becomes 0, add to queue
        4. If result contains all vertices, it's a valid topological sort
           If not, the graph has a cycle
        
        TIME: O(V + E)
        SPACE: O(V)
        
        ADVANTAGE OVER DFS:
        - Naturally detects cycles (result size != V means cycle exists)
        - Iterative approach (no recursion stack limits)
        - More intuitive for understanding dependency resolution
        
        WHY STAFF LEVEL:
        - Can be used to detect cycles
        - Necessary for the "alien dictionary" problem
        - Better for understanding "constraint satisfaction"
        """
        graph = defaultdict(list)
        in_degree = [0] * n
        
        for u, v in edges:
            graph[u].append(v)
            in_degree[v] += 1
        
        queue = deque([i for i in range(n) if in_degree[i] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If not all vertices are in result, graph has a cycle
        if len(result) != n:
            return []  # Cycle detected
        
        return result

    # =========================================================================
    # 2. STRONGLY CONNECTED COMPONENTS (Kosaraju's Algorithm)
    # =========================================================================
    def kosaraju_scc(self, n: int, edges: list) -> list:
        """
        Find all Strongly Connected Components (SCCs) in a directed graph.
        
        EXPLANATION:
        An SCC is a maximal subgraph where every vertex is reachable from
        every other vertex.
        
        Example: 0->1->2->0 forms one SCC (0, 1, 2)
        
        KOSARAJU'S ALGORITHM:
        1. Perform DFS on original graph, record finish times
        2. Reverse the graph (flip all edge directions)
        3. Perform DFS on reversed graph in decreasing order of finish times
        4. Each DFS tree in step 3 is one SCC
        
        WHY IT WORKS:
        - Finish times ensure we process the "boundary" nodes first
        - In the reversed graph, if A can reach B, then B could reach A
        - DFS from a high-finish-time node will only explore its SCC
        
        TIME: O(V + E)
        SPACE: O(V)
        
        REAL-WORLD APPLICATIONS:
        - Social networks: finding friend groups with mutual connections
        - Web graph: identifying strongly connected web pages
        - Circuit design: finding feedback loops
        
        WHY STAFF LEVEL:
        - One of the more complex graph algorithms
        - Deep understanding of graph reversal and finish times
        - Often asked to compare with Tarjan's algorithm
        """
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
        
        # Step 1: DFS and record finish times
        visited = set()
        stack = []
        
        def dfs1(node):
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs1(neighbor)
            stack.append(node)
        
        for i in range(n):
            if i not in visited:
                dfs1(i)
        
        # Step 2: Reverse the graph
        reverse_graph = defaultdict(list)
        for u, v in edges:
            reverse_graph[v].append(u)
        
        # Step 3: DFS on reversed graph in finish-time order
        visited = set()
        sccs = []
        
        def dfs2(node, component):
            visited.add(node)
            component.append(node)
            for neighbor in reverse_graph[node]:
                if neighbor not in visited:
                    dfs2(neighbor, component)
        
        while stack:
            node = stack.pop()
            if node not in visited:
                component = []
                dfs2(node, component)
                sccs.append(component)
        
        return sccs

    # =========================================================================
    # 3. BIPARTITE GRAPH DETECTION
    # =========================================================================
    def is_bipartite(self, n: int, edges: list) -> bool:
        """
        Determine if a graph is bipartite.
        
        EXPLANATION:
        A bipartite graph can be partitioned into two independent sets such that
        every edge connects a vertex from one set to a vertex in the other.
        
        KEY PROPERTY:
        A graph is bipartite if and only if it contains NO ODD-LENGTH CYCLES.
        
        ALGORITHM (Graph Coloring with 2 Colors):
        1. Try to color vertices with 2 colors (0 and 1)
        2. For each unvisited vertex, start BFS
        3. Color it with color 0
        4. For each neighbor, color it with the opposite color
        5. If any neighbor already has the same color, not bipartite
        
        TIME: O(V + E)
        SPACE: O(V)
        
        REAL-WORLD EXAMPLES:
        - Matching problems (job allocation: workers vs jobs)
        - Chess board: coloring squares
        - Social networks: detecting odd-length cycles
        
        WHY STAFF LEVEL:
        - Understanding why odd cycles mean not bipartite
        - Connection to graph coloring problems
        - Use case in matching algorithms
        """
        color = [-1] * n  # -1 = uncolored, 0 = color 1, 1 = color 2
        
        def bfs(start):
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if color[neighbor] == -1:
                        color[neighbor] = 1 - color[node]  # Opposite color
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:  # Same color = odd cycle
                        return False
            
            return True
        
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        for i in range(n):
            if color[i] == -1:
                if not bfs(i):
                    return False
        
        return True

    # =========================================================================
    # 4. GRAPH COLORING (K-Coloring Problem)
    # =========================================================================
    def can_color_graph(self, n: int, edges: list, k: int) -> bool:
        """
        Determine if a graph can be colored using at most k colors such that
        no two adjacent vertices have the same color.
        
        EXPLANATION:
        The graph coloring problem is NP-hard in general, but for small
        values of k and n, we can use backtracking.
        
        ALGORITHM:
        1. Assign colors to vertices one by one
        2. For each vertex, try assigning colors 0 to k-1
        3. Check if the color is safe (not used by any adjacent vertex)
        4. Recursively try to color remaining vertices
        5. If successful, we found a valid coloring
        6. If not, backtrack and try next color
        
        TIME: O(k^V) in worst case
        SPACE: O(V)
        
        OPTIMIZATION:
        - Use heuristics: color vertices with higher degree first
        - Use constraint propagation
        - For bipartite graphs, only 2 colors needed
        
        REAL-WORLD APPLICATIONS:
        - Register allocation (compiler optimization)
        - Sudoku solving
        - Map coloring (geography)
        - Frequency assignment (mobile networks)
        
        WHY STAFF LEVEL:
        - Understanding NP-hard problems
        - Recognizing when backtracking is necessary
        - Trade-off between optimality and computation time
        """
        color = [-1] * n
        graph = defaultdict(list)
        
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        def is_safe(node, color_val):
            for neighbor in graph[node]:
                if color[neighbor] == color_val:
                    return False
            return True
        
        def backtrack(node):
            if node == n:
                return True  # All vertices colored
            
            for c in range(k):
                if is_safe(node, c):
                    color[node] = c
                    if backtrack(node + 1):
                        return True
                    color[node] = -1  # Backtrack
            
            return False
        
        return backtrack(0)

    # =========================================================================
    # 5. ALIEN DICTIONARY (Topological Sort Application)
    # =========================================================================
    def alien_dictionary(self, words: list) -> str:
        """
        Given a list of words sorted in an alien dictionary, derive the
        character ordering of that dictionary.
        
        EXPLANATION:
        This is a classic application of topological sort.
        
        KEY INSIGHT:
        - Compare adjacent words character by character
        - The first differing character gives us an edge in the graph
        - Perform topological sort to find the character ordering
        
        ALGORITHM:
        1. Build a directed graph of characters:
           - For each pair of adjacent words:
             - Find the first differing character
             - If word1[i] != word2[i], add edge word1[i] -> word2[i]
        2. Perform topological sort on this graph
        3. Result is the alien dictionary order
        
        EDGE CASES:
        - Invalid input: "abc" -> "ab" (longer word comes before prefix)
        - No unique ordering possible (cycle in graph)
        
        TIME: O(N * L + U + V) where N = number of words, L = max word length,
              U = unique characters, V = edges
        SPACE: O(U + V)
        
        EXAMPLE:
        Words: ["wrt", "wrf", "er", "ett", "rftt"]
        
        Pairs:
        - "wrt" vs "wrf": w->w, r->r, t->f (edge: t->f)
        - "wrf" vs "er": w->e (edge: w->e)
        - "er" vs "ett": e->e, r->t (edge: r->t)
        - "ett" vs "rftt": e->r (edge: e->r)
        
        Graph: w->e->r->t->f
        Result: "wertf"
        
        WHY STAFF LEVEL:
        - Real topological sort application
        - Handling edge case validation
        - Building graph from constraints
        """
        graph = defaultdict(list)
        in_degree = {}
        
        # Initialize all characters
        for word in words:
            for char in word:
                if char not in in_degree:
                    in_degree[char] = 0
        
        # Build graph from adjacent words
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            
            # Find first differing character
            min_len = min(len(w1), len(w2))
            if len(w1) > len(w2) and w1[:min_len] == w2:
                return ""  # Invalid: longer word comes before prefix
            
            for j in range(min_len):
                if w1[j] != w2[j]:
                    if w2[j] not in graph[w1[j]]:
                        graph[w1[j]].append(w2[j])
                        in_degree[w2[j]] += 1
                    break
        
        # Kahn's algorithm for topological sort
        queue = deque([char for char in in_degree if in_degree[char] == 0])
        result = []
        
        while queue:
            char = queue.popleft()
            result.append(char)
            
            for neighbor in graph[char]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(in_degree):
            return ""  # Cycle detected
        
        return "".join(result)


# =========================================================================
# MAIN EXECUTION BLOCK
# =========================================================================

if __name__ == "__main__":
    sol = AdvancedGraphSolutions()
    
    print("=" * 70)
    print("GRAPH DAY 3: ADVANCED PATTERNS")
    print("=" * 70)
    
    print("\n--- 1. Topological Sort (DFS) ---")
    edges_dag = [(0, 1), (0, 2), (1, 3), (2, 3)]
    topo_dfs = sol.topological_sort_dfs(4, edges_dag)
    print(f"Topological order (DFS): {topo_dfs}")
    print(f"Expected: valid ordering like [0, 1, 2, 3] or [0, 2, 1, 3]")
    
    print("\n--- 2. Topological Sort (Kahn's Algorithm) ---")
    topo_kahn = sol.topological_sort_kahn(4, edges_dag)
    print(f"Topological order (Kahn): {topo_kahn}")
    
    print("\n--- 3. Strongly Connected Components (Kosaraju) ---")
    edges_scc = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3)]
    sccs = sol.kosaraju_scc(5, edges_scc)
    print(f"SCCs: {sccs}")
    print(f"Expected: [[0, 1, 2], [3, 4]] (order may vary)")
    
    print("\n--- 4. Bipartite Graph Detection ---")
    edges_bipartite = [(0, 1), (1, 2), (2, 3), (3, 0)]
    is_bip = sol.is_bipartite(4, edges_bipartite)
    print(f"Is bipartite: {is_bip} (Expected: True)")
    
    edges_not_bipartite = [(0, 1), (1, 2), (2, 0)]
    is_not_bip = sol.is_bipartite(3, edges_not_bipartite)
    print(f"Is bipartite (triangle): {is_not_bip} (Expected: False)")
    
    print("\n--- 5. Graph Coloring (K-Coloring) ---")
    edges_color = [(0, 1), (1, 2), (2, 0)]  # Triangle
    can_color = sol.can_color_graph(3, edges_color, 3)
    print(f"Can color triangle with 3 colors: {can_color} (Expected: True)")
    
    can_color_2 = sol.can_color_graph(3, edges_color, 2)
    print(f"Can color triangle with 2 colors: {can_color_2} (Expected: False)")
    
    print("\n--- 6. Alien Dictionary ---")
    words = ["wrt", "wrf", "er", "ett", "rftt"]
    alien_order = sol.alien_dictionary(words)
    print(f"Alien dictionary order: {alien_order}")
    print(f"Expected: 'wertf' (character ordering)")
    
    print("\n" + "=" * 70)
