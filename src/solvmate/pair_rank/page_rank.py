import numpy as np
from numpy.linalg import norm
from solvmate.pair_rank.pair_rank import DGraph


class ConvergenceError(Exception):
    pass


def page_rank(
    g: DGraph, d: float, eps: float, max_steps: int, nan_on_error=False, norm_prob=True
):
    """
    Computes the page rank for the given directed graph g
    utilizing the damping factor d, the convergence
    threshold eps and a maximum number of steps max_steps.
    Common values for these are:
        d = 0.85 # application dependent!
        eps = 0.1
        max_steps = 200


    >>> d = 0.85; eps = 0.1; max_steps = 200

    A simple case where 1 and 3 reference 2. Hence
    node 2 should show a higher rank:
    >>> g = DGraph()
    >>> g.add_edge(1,2,1)
    >>> g.add_edge(3,2,1)
    >>> page_rank(g,d,eps,100)
    (array([0.21276596, 0.57446809, 0.21276596]), 2)

    In the following example, two nodes just reference
    each other hence should have equal ranks:
    >>> g = DGraph()
    >>> g.add_edge(1,2,1)
    >>> g.add_edge(2,1,1)
    >>> page_rank(g,d,eps,100)
    (array([0.5, 0.5]), 0)

    However, if the first node gives more weight to
    the second than vice versa we get an uneven evaluation:
    >>> g = DGraph()
    >>> g.add_edge(1,2,2)
    >>> g.add_edge(2,1,1)
    >>> page_rank(g,d,eps,100)
    (array([0.60247001, 0.39752999]), 2)

    The above result seemed quite strange at first ...

    But here is the explanation:
    We divide by L in the normal page rank algorithm, because
    every Link counts the same (the edges are not weighted).
    Instead, L is looking at the number of *outgoing* links
    from each node. It correspondingly taxes the connection
    and gives the node where the link points into a *lower score*

    And this is why we see this behavior.

    In other words: The classical pagerank is wholy unsuited
    for our use case as we are interested in the ranking
    of directed graphs with weighted edges.
    Hence, the "link matrix" L should actually have a
    positive contribution to the node ranking.

    I will still leave this proper pagerank implementation
    as a handy reference here. but we will adapt it to our
    use case in a separate function
    """
    N_nodes = len(g.nodes())
    L = g.to_adj_mat().T

    M = np.nan_to_num(1 / L, nan=0.0, posinf=0.0, neginf=0.0)
    assert L.shape[0] == N_nodes

    pr_0 = np.full((N_nodes,), 1 / N_nodes)

    R = pr_0
    Rn = d * M @ R + (1 - d) / N_nodes

    step = 0
    while norm(Rn - R) > eps and step < max_steps:
        R = Rn
        Rn = d * M @ R + (1 - d) / N_nodes
        step += 1

    if step == max_steps:
        if nan_on_error:
            return np.full_like(Rn, np.nan), 0
        else:
            raise ConvergenceError(
                "Could not converge page rank!"
                "Increase num steps or lower eps. Alternatively pass nan_on_error=True to return nans instead"
            )

    if norm_prob:
        Rn = Rn / Rn.sum()
    return Rn, step


def node_rank(
    g: DGraph,
    d: float,
    eps: float,
    max_steps: int,
    min_steps: int = 0,
    nan_on_error=False,
    norm_prob=True,
):
    """
    Computes the node rank for the given directed graph g
    utilizing the damping factor d, the convergence
    threshold eps and a maximum number of steps max_steps.
    Common values for these are:
        d = 0.85 # application dependent!
        eps = 0.1
        max_steps = 200


    >>> d = 0.85; eps = 0.1; max_steps = 200

    A simple case where 1 and 3 reference 2. Hence
    node 2 should show a higher rank:
    >>> g = DGraph()
    >>> g.add_edge(1,2,10)
    >>> g.add_edge(1,3,1)
    >>> node_rank(g,d,eps,100)
    (array([0.25974026, 0.46044864, 0.2798111 ]), 2)

    >>> g = DGraph()
    >>> g.add_edge(1,2,10)
    >>> g.add_edge(2,1,1)
    >>> node_rank(g,d,eps,1000)
    (array([0.35302914, 0.64697086]), 2)

    Let's sanity check everything on a more complex example.

    >>> g = DGraph()
    >>> g.add_edge(1,2,10) # 20 already strongly recommended
    >>> g.add_edge(3,4,2) # 4 only weakly recommended
    >>> g.add_edge(2,5,20) # should win as 2 stronger than 3
    >>> g.add_edge(4,6,20) # 4 weaker hence 6 should loose against 3
    >>> node_rank(g,d,eps,100) # doctest: +NORMALIZE_WHITESPACE
    (array([0.13740955, 0.15987072, 0.13740955, 0.14190178, 0.23128574,
        0.19212267]), 1)

    Interesting question is how much incoming links are accumulated
    vs taking into consideration the hierarchy.
    In the following example, we have 3 nodes all voting for node
    4, but node 4 himself votes for note 5. Should node 5 be
    higher than node 4?
    >>> g = DGraph()
    >>> g.add_edge(1,4,10)
    >>> g.add_edge(2,4,2)
    >>> g.add_edge(3,4,20)
    >>> g.add_edge(4,5,5)
    >>> node_rank(g,d,eps,100) # doctest: +NORMALIZE_WHITESPACE
    (array([0.16850786, 0.16850786, 0.16850786, 0.29238391, 0.2020925 ]), 2)

    So we see that node 4 indeed wins. Let's see what happens when
    node 1 also supports node 5:
    >>> g = DGraph()
    >>> g.add_edge(1,4,10)
    >>> g.add_edge(2,4,2)
    >>> g.add_edge(3,4,20)
    >>> g.add_edge(4,5,5)
    >>> g.add_edge(1,5,20)
    >>> node_rank(g,d,eps,100) # doctest: +NORMALIZE_WHITESPACE
    (array([0.16427911, 0.16427911, 0.16427911, 0.24267195, 0.26449072]), 1)

    We can also advice the node_rank to execute a minimum number of
    iterations:
    >>> g = DGraph()
    >>> g.add_edge(1,4,10)
    >>> g.add_edge(2,4,2)
    >>> g.add_edge(3,4,20)
    >>> g.add_edge(4,5,5)
    >>> g.add_edge(1,5,25)
    >>> node_rank(g,d,eps,100,min_steps=99,) # doctest: +NORMALIZE_WHITESPACE
    (array([0.17006592, 0.17006592, 0.17006592, 0.24467549, 0.24512675]), 99)

    >>> Hex = 0
    >>> MeOH = 1
    >>> EtOH = 2
    >>> H2O = 3
    >>> g = DGraph()
    >>> g.add_edge(Hex,MeOH,6)
    >>> g.add_edge(Hex,EtOH,3)
    >>> g.add_edge(MeOH,H2O,4)
    >>> g.add_edge(EtOH,MeOH,1)
    >>> g.add_edge(EtOH,H2O,2)
    >>> node_rank(g,d,eps,10000,min_steps=100,) # doctest: +NORMALIZE_WHITESPACE
    (array([0.20176655, 0.2785068 , 0.23392309, 0.28580357]), 100)




    """
    N_nodes = len(g.nodes())

    if N_nodes == 0:
        return [], 0

    L = g.to_adj_mat().T

    M = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0) / L.sum()
    assert L.shape[0] == N_nodes

    pr_0 = np.full((N_nodes,), 1 / N_nodes)

    R = pr_0
    Rn = d * M @ R + (1 - d) / N_nodes

    step = 0
    while norm(Rn - R) > eps and step < max_steps or step < min_steps:
        R = Rn
        Rn = d * M @ R + (1 - d) / N_nodes
        step += 1

    if step == max_steps:
        if nan_on_error:
            return np.full_like(Rn, np.nan), 0
        else:
            raise ConvergenceError(
                "Could not converge page rank!"
                "Increase num steps or lower eps. Alternatively pass nan_on_error=True to return nans instead"
            )

    if norm_prob:
        Rn = Rn / Rn.sum()
    return Rn, step
