=============================================================================
STAFF-LEVEL GRAPH REVISION GUIDE: THEORETICAL & ARCHITECTURAL
=============================================================================

--- PART 1: THE "STAFF" ANSWERS TO CORE GRAPH QUESTIONS ---

1. COMPLEXITY ANALYSIS: WHICH ALGORITHM TO USE?

   Problem Type          | Algorithm            | Time Complexity | When to Use
   ==================================================================================
   Shortest Path (Unweighted) | BFS          | O(V+E)         | Weights all = 1
   Shortest Path (Weighted)   | Dijkstra     | O((V+E)logV)   | Non-negative weights
   Shortest Path (Negative)   | Bellman-Ford | O(V*E)         | Negative weights allowed
   All-Pairs Shortest Path    | Floyd-Warshall | O(V^3)       | V ≤ 500, dense graphs
   Cycle Detection (Undirected) | DFS Parent | O(V+E)         | Simple undirected
   Cycle Detection (Directed)   | DFS Colors | O(V+E)         | Directed graphs
   Topological Sort           | Kahn or DFS | O(V+E)         | DAGs, dependencies
   SCCs                       | Kosaraju/Tarjan | O(V+E)     | Strongly connected parts
   Connected Components       | Union-Find | O(α(V)) amortized | Incremental connectivity
   Bipartite Check            | BFS Coloring | O(V+E)        | 2-coloring problem

2. MEMORY BOTTLENECK: GRAPH REPRESENTATION TRADE-OFFS

   Adjacency List:
   - PROS: O(V+E) space (efficient for sparse graphs)
           Easy to iterate neighbors
           Natural for DFS/BFS
   - CONS: O(degree) to check if edge exists
           More pointers = more cache misses

   Adjacency Matrix:
   - PROS: O(1) edge lookup (fast for dense graphs)
   - CONS: O(V²) space (wasteful for sparse graphs)
           Entire row must be scanned for neighbors

   RULE: Use Adjacency List if E ≈ V (sparse), use Matrix if E ≈ V² (dense)

3. THE "PROOF" BEHIND DFS COLORS FOR CYCLE DETECTION (DIRECTED)

   Three Color States:
   - WHITE (0): Not yet visited
   - GRAY (1): Currently being processed (on the recursion stack)
   - BLACK (2): Completely finished processing

   Why GRAY Matters:
   If we encounter a GRAY node during DFS, we've found a back edge
   (edge pointing to an ancestor in the current path). This is a cycle.

   PROOF:
   - When DFS(A) starts, A is marked GRAY
   - DFS recursively explores all neighbors of A
   - If we discover A again while A is GRAY, A is still on the stack
   - This means: A -> ... -> A (a path back to A exists = cycle)

4. TOPOLOGICAL SORT: WHY EXACTLY V-1 ITERATIONS FOR BELLMAN-FORD?

   Claim: A shortest path can have at most V-1 edges.

   PROOF:
   - A path with more than V-1 edges must visit some vertex twice
   - In a cycle-free graph, revisiting = error
   - Therefore, shortest path ≤ V-1 edges

   For Bellman-Ford:
   - Each iteration relaxes one "layer" of the graph
   - After iteration k, all paths of length ≤ k are finalized
   - After V-1 iterations, all shortest paths (max length V-1) are found

5. STRONGLY CONNECTED COMPONENTS: WHY DOES KOSARAJU WORK?

   Key Insight: Finish times in original graph = entry times in reverse graph

   ALGORITHM:
   1. DFS(original graph) → record finish times
   2. DFS(reversed graph) in decreasing finish-time order
   3. Each DFS tree in step 2 = one SCC

   Why Step 3 Finds SCCs:
   - If A and B are in the same SCC, A can reach B and B can reach A
   - In the reversed graph, if A can reach B originally,
     then B can reach A (edges flipped)
   - Processing in reverse finish-time order ensures we find max reachable sets

6. BIPARTITE GRAPHS: ODD CYCLES = NOT BIPARTITE

   Theorem: A graph is bipartite ⟺ it contains NO odd-length cycles

   Proof Direction 1 (Bipartite → No odd cycles):
   - If graph is bipartite, vertices are in set X or Y
   - Every edge connects X to Y
   - Any cycle alternates: X → Y → X → Y → ... → X
   - For cycle to close, length must be even

   Proof Direction 2 (No odd cycles → Bipartite):
   - Try 2-coloring with BFS
   - If successful, graph is bipartite
   - If fails (same color for adjacent nodes), odd cycle exists

--- PART 2: COMMON INTERVIEW PATTERNS ---

PATTERN 1: GRAPH CONSTRUCTION FROM CONSTRAINTS
   Example: Alien Dictionary, Course Prerequisites
   Approach: Model each constraint as an edge
   Then: Run topological sort to find ordering

PATTERN 2: CONNECTED COMPONENTS
   Example: Number of Islands, Friend Circles
   Approach: DFS/BFS from unvisited nodes
   Then: Increment counter for each new component

PATTERN 3: SHORTEST PATH VARIANTS
   Example: Minimum Genetic Mutation, Word Ladder
   Approach: Model as graph, apply BFS or Dijkstra
   Then: Return distance/path from source

PATTERN 4: GRAPH COLORING / PARTITIONING
   Example: Bipartite Matching, Register Allocation
   Approach: Assign states/colors using BFS or backtracking
   Then: Verify constraint satisfaction

--- PART 3: 48-HOUR REVISION CHECKLIST ---

[ ] CODING: Implement all 5 Day 1 problems from scratch
[ ] CODING: Implement Dijkstra and Bellman-Ford (both)
[ ] CODING: Implement Kosaraju's algorithm for SCCs
[ ] VERBAL: Explain why BFS finds shortest path in unweighted graphs
[ ] VERBAL: Explain the "3 colors" approach for directed cycle detection
[ ] ARCHITECTURE: Identify sparse vs dense graphs in given problem
[ ] ARCHITECTURE: Propose Adjacency List vs Matrix for given constraints
[ ] PERFORMANCE: Estimate time/space for large graphs (V=10^6, E=10^7)
[ ] ADVANCED: Solve "Alien Dictionary" using topological sort
[ ] ADVANCED: Solve "Number of Islands" using DFS/Union-Find

--- PART 4: DAY 4 PREVIEW & BEYOND ---

Network Flow (Ford-Fulkerson, Dinic's Algorithm):
   - Max flow problems
   - Bipartite matching
   - Circulation with demands

Planarity & Graph Drawing:
   - Testing if graph can be drawn without edge crossings
   - Kuratowski's theorem

Approximation Algorithms:
   - TSP approximation
   - Vertex cover approximation

Distributed Graph Processing:
   - MapReduce for graph algorithms
   - GraphX, Pregel frameworks

=============================================================================
