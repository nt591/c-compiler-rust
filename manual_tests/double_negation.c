double negate(double x) {
    return -x;
}

int main(void) {
    double result = negate(3.14);
    if (result != -3.14) {
        return 1;
    }
    return 0;
}
