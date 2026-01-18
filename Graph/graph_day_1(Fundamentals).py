# ITERATIVE CYCLE DETECTION FOR UNDIRECTED GRAPH
from collections import deque, defaultdict

def has_cycle_undirected_iterative(n, edges):
    """
    Detect a cycle in an undirected graph using BFS.
    """
    # Build the adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # For undirected graphs

    visited = set()

    # BFS to detect cycle
    for start in range(n):
        if start not in visited:
            queue = deque([(start, -1)])  # Queue stores (node, parent)

            while queue:
                node, parent = queue.popleft()
                visited.add(node)

                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, node))
                    elif neighbor != parent:
                        # If the neighbor is visited and not the parent, it's a cycle
                        return True
    return False

# ITERATIVE CYCLE DETECTION FOR DIRECTED GRAPH
from collections import defaultdict, deque

def has_cycle_directed_topological(n, edges):
    """
    Detect a cycle in a directed graph using Topological Sort (Kahn's Algorithm).
    """
    # Build graph and calculate in-degrees
    graph = defaultdict(list)
    in_degree = {i: 0 for i in range(n)}  # Initialize all nodes with 0 in-degree

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1  # Increment in-degree for the destination node

    # Collect all nodes with in-degree 0
    queue = deque([node for node in range(n) if in_degree[node] == 0])

    visited_count = 0  # Count of visited nodes during topological sort

    while queue:
        node = queue.popleft()
        visited_count += 1

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1  # Remove the edge
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If all nodes are visited, it's a DAG (no cycle), else there's a cycle
    return visited_count != n