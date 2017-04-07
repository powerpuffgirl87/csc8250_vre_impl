
import sys
import random
import networkx as nx

substrate_graph_size = int(sys.argv[1])
substrate_graph_density = float(sys.argv[2])
sgfile = sys.argv[3]
request_graph_size = int(sys.argv[4])
request_graph_density = float(sys.argv[5])
rgfile = sys.argv[6]

def write_graph(gfile, graph):
    with open(gfile, "w") as f:
        gnodes = graph.nodes()
        for n1 in gnodes:
            line = []
            for n2 in gnodes:
                if n1 == n2:
                    line.append(graph.node[n1]["nweight"])
                else:
                    if graph.has_edge(n1, n2):
                        line.append(graph.edge[n1][n2]["eweight"])
                    else:
                        line.append("NA")
            f.write("%s\n" % ",".join(map(str, line)))


node_range = (1, 100)
edge_range = (1, 50)

# generate substrate graph
sg = nx.fast_gnp_random_graph(substrate_graph_size, substrate_graph_density)

for node in sg.nodes():
    sg.node[node]["nweight"] = random.randint(*node_range)

for e1, e2 in sg.edges():
    sg.edge[e1][e2]["eweight"] = random.randint(*edge_range)




# generate request graph
rg = nx.fast_gnp_random_graph(request_graph_size, request_graph_density)

for node in rg.nodes():
    rg.node[node]["nweight"] = random.randint(*node_range)

for e1, e2 in rg.edges():
    rg.edge[e1][e2]["eweight"] = random.randint(*edge_range)


write_graph(rgfile, rg)
write_graph(sgfile, sg)





