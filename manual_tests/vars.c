
int main(void) {
    int i = 2;
    int j = 3;
    int cmp = i < j; // make sure rewrite cmpl j(%rip), i(%rip)

    if (!cmp)
        return 1;
    return 0;
}
