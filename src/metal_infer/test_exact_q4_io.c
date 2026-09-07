#include "engine.h"
#include <stdio.h>

int main(void) {
    const int failure_line = moe_infer_exact_q4_test_run_io_suite();
    if (failure_line != 0) {
        fprintf(stderr, "exact-Q4 I/O self-test failed at engine.c:%d\n", failure_line);
        return 1;
    }
    puts("exact-Q4 I/O self-test: PASS");
    return 0;
}
