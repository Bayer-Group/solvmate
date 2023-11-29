
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

typedef struct
{
    long _cap;
    long len;
    void **data;
} pr_stack;

pr_stack *pr_stack__empty(void)
{
    pr_stack *stack = calloc(1, sizeof(pr_stack));
    stack->_cap = 512;
    stack->len = 0;
    stack->data = calloc(stack->_cap, sizeof(void **));
    return stack;
}

void pr_stack__push(pr_stack *s, void *elt)
{
    long n = s->_cap;
    if (n >= s->_cap)
    {
        s->_cap *= 2;
        s->data = realloc(s->data, s->_cap);
        assert(s->data);
    }

    s->data[s->len] = elt;
    s->len++;
}

void *pr_stack__pop(pr_stack *s)
{
    long n = s->len;
    assert(n != 0);

    void *elt = s->data[n - 1];
    s->len--;
    return elt;
}

int pr_stack__contains(pr_stack *s, void *elt)
{
    void **dat = s->data;
    long i = s->len;
    while (i--)
    {
        if (dat[i] == elt)
        {
            return 1;
        }
    }
    return 0;
}

pr_stack *pr_stack__copy(pr_stack *s)
{
    pr_stack *cpy = pr_stack__empty();

    cpy->data = realloc(cpy->data, s->_cap);
    for (int i = 0; i < s->len; i++)
    {
        cpy->data[i] = s->data[i];
    }
    return cpy;
}