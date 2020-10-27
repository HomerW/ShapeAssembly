def random_dag(nodes, prob):
    # G = nx.gnp_random_graph(nodes, prob, directed=True)
    # dag = nx.DiGraph()
    # dag.add_edges_from([(u, v) for (u, v) in G.edges() if u < v])
    # while nx.is_empty(dag) or (not nx.is_weakly_connected(dag)):
    #     print("empty or disconnected dag, retrying")
    #     G = nx.gnp_random_graph(nodes, prob, directed=True)
    #     dag = nx.DiGraph()
    #     dag.add_edges_from([(u, v) for (u, v) in G.edges() if u < v])
    # dag.remove_nodes_from(list(nx.isolates(dag)))
    # return dag

    max_in_edges = 4
    dag = nx.DiGraph()
    while nx.is_empty(dag) or (not nx.is_weakly_connected(dag)):
        for node in range(nodes):
            neighbors = list(range(nodes))
            random.shuffle(neighbors)
            out_edges = random.choice([0, 1, 2, 3, 4])
            for n in neighbors:
                if len([e for e in dag.edges if node == e[0]]) == out_edges:
                    break
                dag.add_edge(node, n)
                if not nx.algorithms.dag.is_directed_acyclic_graph(dag) or \
                       max([x[1] for x in dag.in_degree(dag.nodes())]) > max_in_edges:
                    dag.remove_edge(node, n)
        max_edges = random.uniform(0.8, 1.2)
        if len(dag.edges) > int(nodes // max_edges):
            print("Too many edges")
            remove_set = random.sample(dag.edges, len(dag.edges)-int(nodes // max_edges))
            dag.remove_edges_from(remove_set)
        dag.remove_nodes_from(list(nx.isolates(dag)))

    return dag

def rand_program(num):
    ALIGN_PROB = 0.75
    ATTACH_PROB = 0.15
    SIZE_MIN = 0.01
    SIZE_MAX = 0.2

    faces = ['right', 'left', 'top', 'bot', 'front', 'back']
    main_count = 0

    def local_to_global_face(local_face, cuboid):
        centroids = []
        global_faces = []
        for face in faces:
            local_centroid, local_ind, local_val = P.getFacePos(face)
            local_centroid = [torch.tensor(x) for x in local_centroid]
            centroids.append(P.cuboids[cuboid].getPos(*local_centroid))
        centroids = torch.stack(centroids)
        right = torch.argmax(centroids, dim=0)
        left = torch.argmin(centroids, dim=0)

    def add_attach_cuboids(c_new, c_old, P, bottom=True):
        counter = 0
        while True:
            if counter > 100:
                print("GIVING UP")
                # if c_new in P.cuboids.keys():
                #     return P
                # else:
                return None

            old_prog = deepcopy(P)

            if c_old == 'bbox':
                if bottom:
                    # bottom of bbox and bottom of (unattached) cuboid
                    attach2 = [random.random(), 0, random.random()]
                    attach1 = [random.random(), 0, random.random()]
                else:
                    # top of bbox and top of (unattached) cuboid
                    attach2 = [random.random(), 1.0, random.random()]
                    attach1 = [random.random(), 1.0, random.random()]
            else:
                face = faces[random.choice(range(6))]
                _, ind, val = P.getFacePos(face)
                # pick random point on random face of first cuboid
                attach1 = [random.random() for _ in range(2)]
                attach1 = attach1[:ind] + [val] + attach1[ind:]
                face = faces[random.choice(range(6))]
                _, ind, val = P.getFacePos(face)
                # pick random point on random face of second cuboid
                attach2 = [random.random() for _ in range(2)]
                attach2 = attach2[:ind] + [val] + attach2[ind:]

            if not c_new in P.cuboids.keys():
                size = [random.uniform(SIZE_MIN, SIZE_MAX) for _ in range(3)]
                alignment = (random.random() < ALIGN_PROB)
                P.execute(f"{c_new} = Cuboid({size[0]}, {size[1]}, {size[2]}, {alignment})")

            P.execute(f"attach({c_new}, {c_old}, "
                      f"{attach1[0]}, {attach1[1]}, {attach1[2]}, "
                      f"{attach2[0]}, {attach2[1]}, {attach2[2]})")
            cuboids_dict = list(P.cuboids.items())
            cuboids_names = [k for k, _ in cuboids_dict if k != 'bbox']
            cuboids = [v for k, v in cuboids_dict if k != 'bbox']
            if len(cuboids) == 1:
                break
            max_overlap = max(inter.findOverlapAmount(cuboids))
            # print(f"{c_new}, {c_old}, {max_overlap}")
            corner_heights = P.cuboids[c_new].getCorners()[:, 1]
            if max_overlap <= 0.15 and (corner_heights >= -0.5).all():
                break
            P = old_prog

            counter += 1

        return P

    num_cuboids = random.choice(range(5, 13))
    dag = random_dag(num_cuboids, ATTACH_PROB)
    print(dag.adj)
    plt.clf()
    nx.draw(dag)
    plt.savefig(f"graph{num}.png")
    cont = True
    while cont:
        main_count += 1
        cont = False
        P = Program()
        queue = [node for node, deg in dag.in_degree(dag.nodes()) if deg == 0]
        sinks = [node for node, deg in dag.out_degree(dag.nodes()) if deg == 0]
        visited = set(queue)
        for v in queue:
            P = add_attach_cuboids(f"cube{v}", 'bbox', P, bottom=True)
            if P is None:
                # cont = True
                # break
                return None
        if cont:
            continue
        # for v in sinks:
        #     P = add_attach_cuboids(f"cube{v}", 'bbox', P, bottom=False)
        #     if P is None:
        #         return None
        while not queue == []:
            v = queue.pop(0)
            for neighbor in dag[v]:
                P = add_attach_cuboids(f"cube{neighbor}", f"cube{v}", P)
                if P is None:
                    # cont = True
                    # break
                    return None
                if not neighbor in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
            if cont:
                break
    return P
