#ifndef CPU_KERNELS_H_
#define CPU_KERNELS_H_

#include <cmath>
#include <stdexcept>
#include <string>
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
void vec_add(A* X, const A* Y, A scale, L N)
{
    static_assert(std::is_floating_point<A>::value,
        "Array should be floating point");
    static_assert(std::is_integral<L>::value, "Array length should be integral");

    for (L i = 0; i < N; ++i)
        X[i] += scale * Y[i];
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
void cpu_reduce_max(A* maxes__bo, L* which__bo, const A* X__to,
    const L* lengths__b, L B, L T, L O)
{
    static_assert(std::is_floating_point<A>::value,
        "Array should be floating point");
    static_assert(std::is_integral<L>::value, "Array length should be integral");

    for (const L* length = lengths__b; length < lengths__b + B; ++length) {
        if (*length == 0)
            continue;
        else if (*length < 0)
            throw std::invalid_argument(std::string("all sequence lengths must be >= 0, was: ") + std::to_string(*length));
        else if (*length > T) {
            throw std::out_of_range("lengths must sum up to the number of rows");
        }

        T -= *length;

        memcpy(maxes__bo, X__to, O * sizeof(maxes__bo[0]));
        X__to += O;
        for (L i = 1; i < *length; ++i) {
            for (L j = 0; j < O; ++j) {
                if (X__to[j] > maxes__bo[j]) {
                    maxes__bo[j] = X__to[j];
                    which__bo[j] = i;
                }
            }
            X__to += O;
        }

        maxes__bo += O;
        which__bo += O;
    }
}

template <typename A, typename L>
void cpu_reduce_mean(A* means__bo, const A* X__to, const L* lengths__b,
    L B, L T, L O)
{
    static_assert(std::is_floating_point<A>::value,
        "Array should be floating point");
    static_assert(std::is_integral<L>::value, "Array length should be integral");

    for (const L* length = lengths__b; length < lengths__b + B; ++length) {
        if (*length == 0)
            continue;
        else if (*length < 0)
            throw std::invalid_argument(std::string("all sequence lengths must be >= 0, was: ") + std::to_string(*length));
        else if (*length > T) {
            throw std::out_of_range("lengths must sum up to the number of rows");
        }

        T -= *length;

        A scale = 1. / *length;
        for (L i = 0; i < *length; ++i) {
            vec_add(means__bo, X__to, scale, O);
            X__to += O;
        }

        means__bo += O;
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
void cpu_reduce_sum(A* sums__bo, const A* X__to, const L* lengths__b,
    L B, L T, L O)
{
    static_assert(std::is_floating_point<A>::value,
        "Array should be floating point");
    static_assert(std::is_integral<L>::value, "Array length should be integral");

    for (const L* length = lengths__b; length < lengths__b + B; ++length) {
        if (*length == 0)
            continue;
        else if (*length < 0)
            throw std::invalid_argument(std::string("all sequence lengths must be >= 0, was: ") + std::to_string(*length));
        else if (*length > T) {
            throw std::out_of_range("lengths must sum up to the number of rows");
        }

        T -= *length;

        for (L i = 0; i < *length; ++i) {
            vec_add(sums__bo, X__to, static_cast<A>(1.0), O);
            X__to += O;
        }

        sums__bo += O;
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
