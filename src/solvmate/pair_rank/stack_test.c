#include "stack.c"

int main(void)
{
    int *pval;
    int val1;
    int val2;
    int val3;
    pr_stack *s;

    s = pr_stack__empty();
    assert(s->len == 0);
    val1 = 123;
    pr_stack__push(s, &val1);
    assert(s->len == 1);

    val2 = 456;
    pr_stack__push(s, &val2);
    assert(s->len == 2);

    val3 = 789;
    pr_stack__push(s, &val3);
    assert(s->len == 3);

    assert(pr_stack__contains(s, &val3));
    pval = pr_stack__pop(s);
    assert(pval == &val3);
    assert(s->len == 2);
    assert(!pr_stack__contains(s, &val3));

    assert(pr_stack__contains(s, &val2));
    pval = pr_stack__pop(s);
    assert(pval == &val2);
    assert(s->len == 1);
    assert(!pr_stack__contains(s, &val2));

    assert(pr_stack__contains(s, &val1));
    pval = pr_stack__pop(s);
    assert(pval == &val1);
    assert(s->len == 0);
    assert(!pr_stack__contains(s, &val1));
}