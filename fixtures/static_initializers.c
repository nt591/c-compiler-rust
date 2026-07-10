/* Initializing static doubles with integer constants and vice versa,
 * exercising implicit-cast insertion and constant folding during
 * semantic analysis. */

double d1 = 2147483647;
double d2 = 4294967295u;
double d3 = 4611686018427389440l;

double uninitialized;

static int i = 4.9;
int unsigned u = 42949.672923E5;
long l = 4611686018427389440.;
unsigned long ul = 18446744073709549568.;

int main(void) {
    return d1 + d2 + d3 + uninitialized + i + u + l + ul > 0;
}
