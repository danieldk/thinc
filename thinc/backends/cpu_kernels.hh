#ifndef CPU_KERNELS_H_
#define CPU_KERNELS_H_

#include <cmath>
#include <type_traits>

template <typename A, typename L>
L argmax(A* arr, L len)
{
    static_assert(std::is_floating_point<A>::value,
        "Array should be floating point");
    static_assert(std::is_integral<L>::value, "Array length should be integral");

    L max = 0;
    for (L i = 1; i < len; ++i) {
        if (arr[i] > arr[max]) {
            max = i;
        }
    }

    return max;
}

template <typename A, typename L>
void cpu_maxout(A* best__bo, L* which__bo, const A* cands__bop, L B, L O, L P)
{
    static_assert(std::is_floating_point<A>::value,
        "Array should be floating point");
    static_assert(std::is_integral<L>::value, "Array length should be integral");

    for (int i = 0; i < B * O; ++i) {
        which__bo[i] = argmax(cands__bop + i * P, P);
        best__bo[i] = cands__bop[i * P + which__bo[i]];
    }
}

template <typename A, typename L>
void cpu_mish(A* Y, L N, A threshold)
{
    static_assert(std::is_floating_point<A>::value,
        "Array should be floating point");
    static_assert(std::is_integral<L>::value, "Array length should be integral");

    for (L i = 0; i < N; ++i) {
        if (Y[i] < threshold) {
            Y[i] *= std::tanh(std::log(1.0 + std::exp(Y[i])));
        }
    }
}

template <typename A, typename L>
void cpu_relu(A* X, L N)
{
    static_assert(std::is_floating_point<A>::value,
        "Array should be floating point");
    static_assert(std::is_integral<L>::value, "Array length should be integral");

    for (L i = 0; i < N; ++i) {
        if (X[i] < 0) {
            X[i] = 0.0;
        }
    }
}

#endif // CPU_KERNELS_H_
