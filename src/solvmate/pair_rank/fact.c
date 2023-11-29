long fact(long n)
{
    if (n < 1)
        return 1;
    return n * fact(n - 1);
}