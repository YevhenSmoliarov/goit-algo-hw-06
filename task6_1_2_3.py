#Створення та візуалізація графа

import networkx as nx
import matplotlib.pyplot as plt

# Створення графа
G = nx.Graph()

# Додавання вузлів
stations = ["A", "B", "C", "D", "E", "F", "G", "H"]
G.add_nodes_from(stations)

# Додавання ребер з вагами (наприклад, відстань між станціями)
edges = [
    ("A", "B", 2), ("A", "C", 5), ("B", "C", 3), ("B", "D", 4),
    ("C", "D", 2), ("C", "E", 6), ("D", "E", 1), ("D", "F", 7),
    ("E", "F", 3), ("F", "G", 8), ("G", "H", 9), ("F", "H", 4)
]
G.add_weighted_edges_from(edges)

# Візуалізація графа
pos = nx.spring_layout(G)
weights = nx.get_edge_attributes(G, 'weight')

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=15, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
plt.title("Транспортна мережа міста")
plt.show()

#Аналіз основних характеристик графа

# Кількість вершин
num_nodes = G.number_of_nodes()
print(f"Кількість вершин: {num_nodes}")

# Кількість ребер
num_edges = G.number_of_edges()
print(f"Кількість ребер: {num_edges}")

# Ступінь вершин
degrees = dict(G.degree())
print(f"Ступінь кожної вершини: {degrees}")

#Реалізація алгоритмів DFS та BFS

def dfs_path(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in set(graph.neighbors(vertex)) - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))

def bfs_path(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in set(graph.neighbors(vertex)) - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

# Приклад використання
start_node = "A"
goal_node = "H"

print("DFS шляхи:")
dfs_paths = list(dfs_path(G, start_node, goal_node))
for path in dfs_paths:
    print(path)

print("BFS шляхи:")
bfs_paths = list(bfs_path(G, start_node, goal_node))
for path in bfs_paths:
    print(path)

#Алгоритм Дейкстри

def dijkstra(graph, start):
    shortest_paths = {start: (None, 0)}
    current_node = start
    visited = set()

    while current_node:
        visited.add(current_node)
        destinations = graph.edges(current_node)
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node, weight in graph[current_node].items():
            weight = weight['weight'] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return shortest_paths

        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    return shortest_paths

def shortest_path(graph, origin, destination):
    shortest_paths = dijkstra(graph, origin)
    path = []
    while destination:
        path.append(destination)
        next_node = shortest_paths[destination][0]
        destination = next_node
    path = path[::-1]
    return path

# Приклад використання
origin = "A"
destination = "H"
shortest_path_result = shortest_path(G, origin, destination)
print(f"Найкоротший шлях з {origin} до {destination}: {shortest_path_result}")