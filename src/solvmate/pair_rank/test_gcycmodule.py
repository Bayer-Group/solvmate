import gcyc


def test_gcyc_on_acyclic_graph():
    nodes = [0, 1, 2]
    edges = [[0, 1], [1, 2], [0, 2]]
    assert not gcyc.dgraph_contains_cycle(nodes, edges)


def test_gcyc_on_cyclic_graph():
    nodes = [0, 1, 2]
    edges = [[0, 1], [1, 2], [2, 0]]
    assert gcyc.dgraph_contains_cycle(nodes, edges)


def test_gcyc_on_complicated_graph():
    nodes = [0, 1, 2, 3, 4]
    edges = [[1, 2], [1, 4], [2, 4], [2, 3], [0, 1], [3, 0]]
    assert gcyc.dgraph_contains_cycle(nodes, edges)
