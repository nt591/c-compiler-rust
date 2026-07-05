double half(void) {
    return 0.5;
}

int main(void) {
    double result = half();
    if (result != 0.5) {
        return 1;
    }
    return 0;
}
