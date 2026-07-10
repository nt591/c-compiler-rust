int is_valid(int x, int y) {
    return x > 0 && y > 0;
}

int classify(int x, int y) {
    int total = 0;
    int i = 0;

    if (!is_valid(x, y)) {
        return -1;
    }

    while (i < x) {
        if (i % 2 == 0) {
            total = total + i;
        } else {
            total = total - i;
        }
        i = i + 1;
    }

    for (int j = 0; j < y; j = j + 1) {
        total = total + j;
        if (total > 100) {
            break;
        }
    }

    return total >= 0 || total == -1;
}
