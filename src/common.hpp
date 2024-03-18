// common stuff

#pragma once

#include <cstdio>

static inline void
cuda_check(cudaError_t err, const char *who = nullptr)
{
    if (err == cudaSuccess)
        return;

    fprintf(stderr,
            "[cuda error] (%s): %s\n",
            who ? who : "?",
            cudaGetErrorString(err));
    exit(1);
}
