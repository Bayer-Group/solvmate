#include "Python.h"
#include <stdio.h>
#include <stdlib.h>
#include "graph.c"

#define MAX_NODES 1024
#define MAX_EDGES (MAX_NODES * MAX_NODES)

void my_assert(int cond, char *msg)
{
    if (!cond)
    {
        puts(msg);
        puts("");
        exit(1);
    }
}

static PyObject *gcyc_say_hello(PyObject *self, PyObject *args)
{
    printf("Hello World\n");

    Py_RETURN_NONE;
}

static PyObject *gcyc_dgraph_contains_cycle(PyObject *self, PyObject *args)
{
    PyObject *nodes, *edges;
    if (!PyArg_ParseTuple(args, "OO", &nodes, &edges))
    {
        return NULL;
    }
    my_assert(PyList_Size(nodes) >= 0, "nodes list corrupt");
    int n_nodes = PyList_Size(nodes);
    int n_edges = PyList_Size(edges);

    my_assert(n_nodes < MAX_NODES, "max nodes exceeded");
    my_assert(n_edges < MAX_EDGES, "max edges exceeded");

    pr_graph *g = pr_graph__make(n_nodes);

    for (int i = 0; i < n_edges; i++)
    {
        PyObject *e = PyList_GetItem(edges, i);
        my_assert(PyList_Size(e) == 2, "edge must contain exactly two nodes"); /* Any edge connects two nodes */
        PyObject *from = PyList_GetItem(e, 0);
        PyObject *to = PyList_GetItem(e, 1);
        my_assert(PyLong_Check(from), "edge from index is not an integer");
        my_assert(PyLong_Check(to), "edge to index is not an integer");
        int ifrom = (int)PyLong_AsLong(from);
        int ito = (int)PyLong_AsLong(to);
        pr_graph__set_connected(g, ifrom, ito, 1);
    }

    int contains_cycle = pr_graph__contains_cycle_any(g);

    // cleanup
    pr_graph__destroy(g);

    PyObject *rslt = Py_BuildValue("i", contains_cycle);

    return rslt;
}

static PyMethodDef GCycMethods[] = {
    {"say_hello", gcyc_say_hello, METH_VARARGS, "Say Hello World"},
    {"dgraph_contains_cycle", gcyc_dgraph_contains_cycle, METH_VARARGS,
     "1 if dgraph contains cycle, 0 otherwise."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef gcyc =
    {
        PyModuleDef_HEAD_INIT,
        "gcyc",
        "",
        -1,
        GCycMethods};

PyMODINIT_FUNC PyInit_gcyc(void)
{
    PyObject *module;
    module = PyModule_Create(&gcyc);

    if (module == NULL)
        return NULL;

    return module;
}
