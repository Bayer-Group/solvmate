#include "graph.c"

// Looks at the graph of the form 0--1--2--3
// and continues to add/remove edges while
// checking that resulting cycles are properly
// detected iff present.
void simple_4_node_graph(void);

// Looks at the graph of the form 0--1--2--4--5
//                                 \        \6/
//                                  3
// and continues to add/remove edges while
// checking that resulting cycles are properly
// detected iff present.
// Initially, we have the cycle (4,5,6) which
// should be detected.
//
// We then proceed by removing either 4,6 or 5,6
// and we see that cycles are gone.
//
// Then, we add a connections 2,3 and 3,0 which again
// introduces a cycle.
void branched_7_node_graph(void);

void complicated_graph(void);

// An example where the cycle cannot be reached
// from the initial node.
void not_reachable_from_initial_node(void);

int main(int argc, char **args)
{
    simple_4_node_graph();
    branched_7_node_graph();
    complicated_graph();
    not_reachable_from_initial_node();
    return 0;
}

void simple_4_node_graph(void)
{
    pr_graph *g = pr_graph__make(4);
    // a graph connected from 1,2,3 linearly
    // so no cycles!
    pr_graph__set_connected(g, 0, 1, 1);
    pr_graph__set_connected(g, 1, 2, 1);
    pr_graph__set_connected(g, 2, 3, 1);
    int *rec_stack = calloc(g->n_nodes, sizeof(int));
    int *exhausted = calloc(g->n_nodes, sizeof(int));
    assert(!pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    // Now we close the loop by connecting
    // the nodes 1 and 3: We have a cycle!
    pr_graph__set_connected(g, 3, 0, 1);

    assert(pr_graph__is_connected(g, 3, 0));
    assert(pr_graph__is_connected(g, 1, 2));
    assert(!pr_graph__is_connected(g, 0, 3));
    assert(!pr_graph__is_connected(g, 0, 2));
    memset(rec_stack, 0, g->n_nodes * sizeof(int));
    memset(exhausted, 0, g->n_nodes * sizeof(int));
    assert(pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    // Let's Remove the cycle by removing the
    // edge 1->2
    pr_graph__set_connected(g, 3, 0, 1);

    memset(rec_stack, 0, g->n_nodes * sizeof(int));
    memset(exhausted, 0, g->n_nodes * sizeof(int));
    pr_graph__set_connected(g, 1, 2, 0);
    assert(!pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    // Cleanup
    pr_graph__destroy(g);
    free(rec_stack);
    free(exhausted);
}

void branched_7_node_graph(void)
{
    // Looks at the graph of the form 0--1--2--4--5
    //                                 \        \6/
    //                                  3
    pr_graph *g = pr_graph__make(7);

    pr_graph__set_connected(g, 0, 1, 1);
    pr_graph__set_connected(g, 1, 2, 1);
    pr_graph__set_connected(g, 0, 3, 1);
    pr_graph__set_connected(g, 2, 4, 1);
    pr_graph__set_connected(g, 4, 5, 1);
    pr_graph__set_connected(g, 5, 6, 1);
    pr_graph__set_connected(g, 6, 4, 1);
    int *rec_stack = calloc(g->n_nodes, sizeof(int));
    int *exhausted = calloc(g->n_nodes, sizeof(int));
    assert(pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    // We then proceed by removing either 6,4 or 5,6
    // and we see that cycles are gone.
    pr_graph__set_connected(g, 6, 4, 0);
    memset(rec_stack, 0, g->n_nodes * sizeof(int));
    memset(exhausted, 0, g->n_nodes * sizeof(int));
    assert(!pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    pr_graph__set_connected(g, 6, 4, 1);
    pr_graph__set_connected(g, 5, 6, 0);
    memset(rec_stack, 0, g->n_nodes * sizeof(int));
    memset(exhausted, 0, g->n_nodes * sizeof(int));
    assert(!pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    // Then, we add a connection 2,3 and 3,0 which again
    // introduces a cycle.
    pr_graph__set_connected(g, 2, 3, 1);
    pr_graph__set_connected(g, 3, 0, 1);
    memset(rec_stack, 0, g->n_nodes * sizeof(int));
    memset(exhausted, 0, g->n_nodes * sizeof(int));
    assert(pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    // However, unsetting 3,0 would again render the graph
    // to be acyclic, as the edge 0,3 cannot be traversed
    // in the wrong direction:
    pr_graph__set_connected(g, 2, 3, 1);
    pr_graph__set_connected(g, 3, 0, 0);
    memset(rec_stack, 0, g->n_nodes * sizeof(int));
    memset(exhausted, 0, g->n_nodes * sizeof(int));
    assert(!pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    // Cleanup
    pr_graph__destroy(g);
    free(rec_stack);
    free(exhausted);
}

void complicated_graph(void)
{
    pr_graph *g = pr_graph__make(5);
    pr_graph__set_connected(g, 1, 2, 1);
    pr_graph__set_connected(g, 1, 4, 1);
    pr_graph__set_connected(g, 2, 4, 1);
    pr_graph__set_connected(g, 2, 3, 1);
    pr_graph__set_connected(g, 0, 1, 1);
    pr_graph__set_connected(g, 3, 0, 1);
    int *rec_stack = calloc(g->n_nodes, sizeof(int));
    int *exhausted = calloc(g->n_nodes, sizeof(int));
    assert(pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));
}

void not_reachable_from_initial_node(void)
{
    pr_graph *g = pr_graph__make(5);
    pr_graph__set_connected(g, 2, 1, 1);
    pr_graph__set_connected(g, 3, 2, 1);
    pr_graph__set_connected(g, 4, 3, 1);
    pr_graph__set_connected(g, 2, 4, 1);
    assert(pr_graph__contains_cycle_any(g));

    // however, the following does not work ...
    int *rec_stack = calloc(g->n_nodes, sizeof(int));
    int *exhausted = calloc(g->n_nodes, sizeof(int));
    // This graph actually contains a cycle but it cannot be reached
    // from the initial node...
    assert(!pr_graph__contains_cycle_rec(0, g, rec_stack, exhausted));

    // From node 2 it would work:
    assert(pr_graph__contains_cycle_rec(2, g, rec_stack, exhausted));
}