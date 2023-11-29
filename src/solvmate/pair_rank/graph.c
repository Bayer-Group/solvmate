#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include "stack.c"

#define PR_VERBOSE_GRAPH 0

#define INITIAL_CAP_NODES 1024

typedef struct
{
    long n_nodes;
    int *adj;
} pr_graph;

pr_graph *pr_graph__make(long n)
{
    pr_graph *g = calloc(1, sizeof(pr_graph));
    g->n_nodes = n;
    g->adj = calloc(n * n, sizeof(int));
    return g;
}

void pr_graph__destroy(pr_graph *g)
{
    free(g->adj);
    free(g);
}

int pr_graph__at(pr_graph *g, long i, long j)
{
    return (g->n_nodes * i + j);
}

int pr_graph__is_connected(pr_graph *g, long i, long j)
{
    long at = pr_graph__at(g, i, j);
    return g->adj[at];
}

void pr_graph__set_connected(pr_graph *g, long i, long j, int connected)
{
    long at = pr_graph__at(g, i, j);
    g->adj[at] = connected;
}

int pr_graph__contains_cycle_rec(
    // Naive Recursive implementation of cycle detection
    // within a directed graph.
    // Usage:
    // ------
    int i,
    // -- The node from where to start the cycle detection
    //    and the current node in the recursive call.
    pr_graph *g,
    // -- Pointer to the directed graph.
    int *rec_stack,
    // -- An array holding the already computed
    //    recursive subproblem solutions.
    int *exhausted
    // -- An array holding the already travelled
    //    nodes that should not be visited again.
    //
    // Usage:
    // ------
    // See graph_test.c/simple_4_node_graph()
)
{
#if PR_VERBOSE_GRAPH
    printf("i = %d\n", i);
    for (int q = 0; q < g->n_nodes; q++)
    {
        printf("rec_stack[%d] = %d\n", q, rec_stack[q]);
    }
    for (int q = 0; q < g->n_nodes; q++)
    {
        printf("exhausted[%d] = %d\n", q, exhausted[q]);
    }
#endif
    if (rec_stack[i])
        return 1; /* found cycle */
    if (exhausted[i])
        return 0; /* already checked before*/

    exhausted[i] = 1;
    rec_stack[i] = 1;

    for (int j = 0; j < g->n_nodes; j++)
    {
        if (pr_graph__is_connected(g, i, j))
        {
            int rec_result = pr_graph__contains_cycle_rec(j, g, rec_stack, exhausted);
            if (rec_result)
                return 1;
        }
    }

    rec_stack[i] = 0;

    return 0;
}

int pr_graph__contains_cycle_any(pr_graph *g)
{
    int cycle = 0;
    int *rec_stack = calloc(g->n_nodes, sizeof(int));
    int *exhausted = calloc(g->n_nodes, sizeof(int));

    for (int start = 0; start < g->n_nodes; start++)
    {
        if (pr_graph__contains_cycle_rec(start, g, rec_stack, exhausted))
        {
            cycle = 1;
            goto finalize;
        }
        // Remember to reset the internal state of
        // the cycle search.
        memset(rec_stack, 0, g->n_nodes * sizeof(int));
        memset(exhausted, 0, g->n_nodes * sizeof(int));
    }
    goto finalize; // TODO: remove as should be redundant

finalize:
    free(rec_stack);
    free(exhausted);
    return cycle;
}