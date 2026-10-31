#include "spark_mlm_runtime.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: spark_mlm_demo index.smlm 64hex-signature\n");
        return 2;
    }
    std::array<uint8_t, 32> sig{};
    const char *hex = argv[2];
    for (size_t i = 0; i < 32; ++i) {
        int hi = 0, lo = 0;
        if (!hex[2*i] || !hex[2*i+1] ||
            std::sscanf(hex + 2*i, "%1x%1x", &hi, &lo) != 2) return 2;
        sig[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    if (hex[64] != '\0') return 2;
    spark_mlm_index_t *model = nullptr;
    const int status = spark_mlm_index_open_v1(argv[1], &model);
    if (status < 0) { std::fprintf(stderr, "index open: %d\n", status); return 3; }
    int32_t tokens[64], scores[64];
    const int count = spark_mlm_probe_v1(
        reinterpret_cast<uintptr_t>(model), reinterpret_cast<uintptr_t>(sig.data()), 64,
        reinterpret_cast<uintptr_t>(tokens), reinterpret_cast<uintptr_t>(scores));
    spark_mlm_index_close_v1(model);
    if (count < 0) { std::fprintf(stderr, "probe: %d\n", count); return 4; }
    std::printf("count=%d", count);
    for (int i = 0; i < count; ++i) std::printf(" token=%d:%d", tokens[i], scores[i]);
    std::printf("\n");
    return 0;
}
