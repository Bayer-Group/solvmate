from collections import defaultdict
from copy import deepcopy
import numpy as np

try:
    import gcyc

    _FAST_C_IMPL = True
except:
    _FAST_C_IMPL = False
    print(
        """
        Could not import gcyc module. That means the fast c implementation of dgraph cycle detection is not available
        It is recommended to install the fast cycle detection by going into the pair_rank folder and running `make all`
        which will both install the c extension and run appropriate tests to assure a sane build.
        """
    )
    assert False


class DGraph:
    """
    A simple implementation of a directed graph. No optimizations applied
    and only handles a few thousand cases (due to call stack limitations
    in the recursive implementation).

    But that should be efficient enough for our use case of ~hundred nodes
    and ~hundred edges.

    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(2,3,None)
    >>> dg.contains_cycle()
    False
    >>> dg.add_edge(3,1,None)
    >>> dg.contains_cycle()
    True
    >>> dg.add_edge(4,5,None)
    >>> dg.contains_cycle()
    True
    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(2,1,None)
    >>> dg.contains_cycle()
    True
    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(1,3,None)
    >>> dg.add_edge(1,4,None)
    >>> dg.add_edge(4,5,None)
    >>> dg.contains_cycle()
    False
    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(1,3,None)
    >>> dg.add_edge(1,4,None)
    >>> dg.add_edge(4,5,None)
    >>> dg.add_edge(5,3,None)
    >>> dg.contains_cycle()
    False
    >>> dg.find_source()
    [1]

    Finally, lets test a larger cycle:
    >>> dg = DGraph()
    >>> dg.add_edge(2,3,None)
    >>> dg.add_edge(3,4,None)
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(4,1,None)
    >>> dg.contains_cycle()
    True

    >>> dg = DGraph()
    >>> dg.add_edge(2,3,None)
    >>> dg.add_edge(2,5,None)
    >>> dg.add_edge(3,5,None)
    >>> dg.add_edge(3,4,None)
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(4,1,None)
    >>> dg.contains_cycle()
    True

    The graph
    1 -> 2 -> 3 -> 4
             ^    /
             |  v
               5
    contains a cycle (3,4,5):
    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(2,3,None)
    >>> dg.add_edge(3,4,None)
    >>> dg.add_edge(4,5,None)
    >>> dg.add_edge(5,3,None)
    >>> dg.contains_cycle()
    True

    Same if we reverse the order:
    >>> dg = DGraph()
    >>> dg.add_edge(2,1,None)
    >>> dg.add_edge(3,2,None)
    >>> dg.add_edge(4,3,None)
    >>> dg.add_edge(5,4,None)
    >>> dg.add_edge(3,5,None)
    >>> dg.contains_cycle()
    True

    However, once we remove the edge 4,5
    it no longer contains a cycle:
    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(2,3,None)
    >>> dg.add_edge(3,4,None)
    >>> dg.add_edge(5,3,None)
    >>> dg.contains_cycle()
    False

    Let's look at a slightly more
    idiopathic example that contains a
    cycle which is disjointed from the rest:
    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(2,3,None)
    >>> dg.add_edge(3,4,None)
    >>> dg.add_edge(5,6,None)
    >>> dg.add_edge(6,7,None)
    >>> dg.add_edge(7,5,None)
    >>> dg.contains_cycle()
    True

    An example containing a cycle with heavy
    branching before:
    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(2,3,None)
    >>> dg.add_edge(2,4,None)
    >>> dg.add_edge(2,5,None)
    >>> dg.add_edge(3,4,None)
    >>> dg.add_edge(5,6,None)
    >>> dg.add_edge(6,7,None)
    >>> dg.add_edge(7,5,None)
    >>> dg.contains_cycle()
    True

    >>> dg = DGraph()
    >>> dg.add_edge(1,2,None)
    >>> dg.add_edge(2,3,None)
    >>> dg.add_edge(2,4,None)
    >>> dg.add_edge(2,5,None)
    >>> dg.add_edge(3,4,None)
    >>> dg.add_edge(5,6,None)
    >>> dg.add_edge(6,7,None)
    >>> dg.add_edge(5,7,None)
    >>> dg.contains_cycle()
    False

    """

    def __init__(self, use_fast_c_impl=True) -> None:
        self.edge_start: dict[int, list[int]] = defaultdict(list)
        self.edge_end: dict[int, list[int]] = defaultdict(list)
        self.edge_annot: dict[(int, int), any] = {}
        self.use_fast_c_impl = use_fast_c_impl

    def to_adj_mat(
        self,
    ) -> np.ndarray:
        """
        Returns the adjacency matrix of the given directed graph.
        >>> g = DGraph()
        >>> g.add_edge(1,2,0.5)
        >>> g.add_edge(2,3,1.5)
        >>> g.add_edge(3,4,2.5)
        >>> g.to_adj_mat() # doctest: +NORMALIZE_WHITESPACE
        array([[0. , 0.5, 0. , 0. ],
        [0. , 0. , 1.5, 0. ],
        [0. , 0. , 0. , 2.5],
        [0. , 0. , 0. , 0. ]])
        """
        nodes = sorted(self.nodes())
        n_to_i = {n: i for i, n in enumerate(nodes)}
        i_to_n = {i: n for i, n in enumerate(nodes)}

        nodes_i = [n_to_i[n] for n in nodes]
        edges_i = [
            [n_to_i[start], n_to_i[end]]
            for start, ends in self.edge_start.items()
            for end in ends
        ]

        m = np.zeros((len(nodes_i), len(nodes_i)))
        for node_start in nodes_i:
            for edge in edges_i:
                if edge[0] == node_start:
                    node_end = edge[1]
                    m[node_start, node_end] = self.edge_annot[
                        (i_to_n[node_start], i_to_n[node_end])
                    ]
        return m

    def copy(self) -> "DGraph":
        copy = DGraph()
        copy.edge_start.update(deepcopy(self.edge_start))
        copy.edge_end.update(deepcopy(self.edge_end))
        copy.edge_annot.update(deepcopy(self.edge_annot))
        return copy

    def nodes(
        self,
    ):
        return sorted(set(list(self.edge_start.keys()) + list(self.edge_end.keys())))

    def with_nodes_removed(self, nodes_to_remove: list[int]) -> "DGraph":
        rslt = DGraph()
        for i, j in self.edge_start.items():
            if i not in nodes_to_remove and j not in nodes_to_remove:
                rslt.edge_start[i] = j
        for i, j in self.edge_end.items():
            if i not in nodes_to_remove and j not in nodes_to_remove:
                rslt.edge_start[i] = j

        for (i, j), annot in self.edge_annot.items():
            if i not in nodes_to_remove and j not in nodes_to_remove:
                rslt.edge_annot[(i, j)] = annot
        return rslt

    def find_source(self):
        source_nodes = [node for node in self.edge_start if node not in self.edge_end]
        return source_nodes

    def add_edge(self, i, j, annot):
        self.edge_start[i].append(j)
        self.edge_end[j].append(i)
        self.edge_annot[(i, j)] = annot

    def contains_cycle(self):
        if self.use_fast_c_impl:
            return self._contains_cycle_c()
        else:
            return self._contains_cycle_py()

    def _contains_cycle_c(self):
        assert (
            _FAST_C_IMPL
        ), "fast c implementation not available. must install gcyc c extension first!"
        nodes = sorted(self.nodes())
        n_to_i = {n: i for i, n in enumerate(nodes)}

        nodes_i = [n_to_i[n] for n in nodes]
        edges_i = [
            [n_to_i[start], n_to_i[end]]
            for start, ends in self.edge_start.items()
            for end in ends
        ]

        return bool(gcyc.dgraph_contains_cycle(nodes_i, edges_i))

    def _contains_cycle_py(self):
        all_visited_nodes = {}
        all_visited_edges = {}
        nodes = self.nodes()
        while len(all_visited_nodes) != len(nodes):
            v = [node for node in nodes if node not in all_visited_nodes][0]
            this_visited_nodes = {}
            this_visited_edges = {}
            if self._contains_cycle_from_node(
                v,
                this_visited_nodes,
                this_visited_edges,
                max_depth=10000,
            ):
                return True
            else:
                all_visited_nodes.update(this_visited_nodes)
                all_visited_edges.update(this_visited_edges)
        return False

    def _contains_cycle_from_node(
        self,
        v,
        visited_nodes,
        visited_edges,
        max_depth: int,
        this_depth=0,
    ):
        if this_depth >= max_depth:
            raise Exception("max depth in cycle detection exceeded!")
        if v in visited_nodes:
            return True
        for to in self.edge_start[v]:
            if (v, to) in visited_edges:
                continue
            else:
                visited_nodes[v] = True
                visited_edges[(v, to)] = True
                leap_success = self._contains_cycle_from_node(
                    to,
                    dict(visited_nodes),
                    dict(visited_edges),
                    max_depth=max_depth,
                    this_depth=this_depth + 1,
                )
                if leap_success:
                    return True
        visited_nodes[v] = True
        return False


class PairRank:
    """
    Implements the RankedPair algorithm.
    A description of the algorithm can be found here:
    https://en.wikipedia.org/wiki/Ranked_pairs

    In the following example node-1 is 50$ stronger than node-2
    which in turn is 30$ stronger than node-3.
    Hence, we find node-1 as the winning node.
    >>> g_in = DGraph()
    >>> g_in.add_edge(1,2,50) # here one is stronger than 2!
    >>> g_in.add_edge(2,3,30)
    >>> pr = PairRank()
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [1]

    Same example but with flipped order of edge addition
    >>> g_in = DGraph()
    >>> g_in.add_edge(2,3,30)
    >>> g_in.add_edge(1,2,50) # here one is stronger than 2!
    >>> pr = PairRank()
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [1]

    This time we add a fourth node that is slightly stronger
    than node-1 and should therefore win overall:
    >>> g_in = DGraph()
    >>> g_in.add_edge(2,3,30)
    >>> g_in.add_edge(1,2,50)
    >>> g_in.add_edge(4,1,1) # just slightly better but enough!
    >>> pr = PairRank()
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [4]

    Let's add another node that is not changing anything:
    >>> g_in = DGraph()
    >>> g_in.add_edge(2,3,30)
    >>> g_in.add_edge(1,2,50)
    >>> g_in.add_edge(3,5,5000)
    >>> g_in.add_edge(4,1,1) # just slightly better but enough!
    >>> pr = PairRank()
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [4]

    Finally, a more complicated example. Here the node 4 gets
    a stronger competing node which we attached to node-2 but
    made it that much stronger:
    >>> g_in = DGraph()
    >>> g_in.add_edge(2,3,30)
    >>> g_in.add_edge(1,2,50)
    >>> g_in.add_edge(4,1,1)
    >>> g_in.add_edge(5,2,60) # 5 is stronger than 1
    >>> g_in.add_edge(6,5,2) # 6 is stronger than 4
    >>> pr = PairRank()
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [6, 4]

    As we saw before, in those cases where there are several
    graph sources, the find_source routine will return several
    nodes. Here we make sure that we do actually get the strongest
    node in such a case:
    >>> g_in = DGraph()
    >>> g_in.add_edge(2,3,30)
    >>> g_in.add_edge(1,2,50)
    >>> g_in.add_edge(4,1,1)
    >>> g_in.add_edge(5,2,60) # 5 is stronger than 1
    >>> g_in.add_edge(6,5,2) # 6 is stronger than 4
    >>> g_in.add_edge(7,5,3) # 7 is strongest
    >>> g_in.add_edge(8,5,2.5) # 8 is second-strongest
    >>> pr = PairRank()
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [7, 8, 6, 4]

    As expected, we get the source nodes in decreasing order
    of relevance. Often, we also would like to find out which
    "non-source" nodes might be the most relevant.
    Aka after having acknowledged node-7 as the leader, we
    might be interested in which node is the second best.
    For that we procede in the following way:
    >>> g_in = g_in.with_nodes_removed([7])
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [8, 6, 4]
    >>> g_in = g_in.with_nodes_removed([8])
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [6, 4]
    >>> g_in = g_in.with_nodes_removed([6])
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [5, 4]

    And we notice that five is actually a stronger node but
    was hidden as it was not a source node before.

    Let's test an example that is containing a cycle. Here,
    2 is stronger than 1 while 3 is stronger than 2 and
    1 is stronger than 3. As node-1 is the one with the
    weakest link he will be ignored. Therefore, 3 will
    win the race:
    >>> g_in = DGraph()
    >>> g_in.add_edge(2,1,20)
    >>> g_in.add_edge(3,2,30)
    >>> g_in.add_edge(1,3,10)
    >>> g_in.contains_cycle()
    True
    >>> pr = PairRank()
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [3]

    After that come nodes 2 and 1 in that order:
    >>> g_in = g_in.with_nodes_removed([3])
    >>> g_out = pr.run(g_in)
    >>> g_out.contains_cycle()
    False
    >>> g_out.find_source()
    [2]


    """

    def __init__(
        self,
    ) -> None:
        pass

    def run(
        self,
        g_input: DGraph,
    ):
        # Iterate over all pairs starting with the largest difference
        filtered_edges = [kv for kv in g_input.edge_annot.items()]
        edges_by_strength = sorted(
            filtered_edges,
            key=lambda kv: -kv[1],
        )

        g_final = DGraph()

        for (start, end), strength in edges_by_strength:
            g_trial = g_final.copy()
            g_trial.add_edge(start, end, strength)
            if g_trial.contains_cycle():
                continue
            else:
                g_final.add_edge(start, end, strength)

        return g_final
