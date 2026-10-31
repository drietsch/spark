#include "spark_semantic_runtime.h"
#include "spark_mlm_runtime.h"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "usage: spark_semantic_demo encoder.senc index.smlm mask_index token_id ...\n";
        return 2;
    }
    spark_semantic_encoder_t *encoder = nullptr;
    spark_mlm_index_t *index = nullptr;
    if (spark_semantic_encoder_open_v1(argv[1], &encoder) != 0 ||
        spark_mlm_index_open_v1(argv[2], &index) != 0) {
        std::cerr << "failed to open encoder or index\n";
        spark_semantic_encoder_close_v1(encoder);
        spark_mlm_index_close_v1(index);
        return 1;
    }
    try {
        int mask = std::stoi(argv[3]);
        std::vector<uint32_t> tokens;
        for (int i = 4; i < argc; ++i) {
            const auto v = std::stoull(argv[i]);
            if (v > UINT32_MAX) throw std::out_of_range("token id");
            tokens.push_back(static_cast<uint32_t>(v));
        }
        std::array<int32_t, 64> ids{}, scores{};
        int n = spark_semantic_predict_v1(reinterpret_cast<uint64_t>(encoder),
                                           reinterpret_cast<uint64_t>(index),
                                           reinterpret_cast<uint64_t>(tokens.data()),
                                           static_cast<int>(tokens.size()), mask, 64,
                                           reinterpret_cast<uint64_t>(ids.data()),
                                           reinterpret_cast<uint64_t>(scores.data()));
        if (n < 0) {
            std::cerr << "prediction failed, status=" << n << "\n";
            spark_semantic_encoder_close_v1(encoder);
            spark_mlm_index_close_v1(index);
            return 1;
        }
        std::cout << "count=" << n << " token=" << (n ? ids[0] : -1) << "\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        spark_semantic_encoder_close_v1(encoder);
        spark_mlm_index_close_v1(index);
        return 2;
    }
    spark_semantic_encoder_close_v1(encoder);
    spark_mlm_index_close_v1(index);
    return 0;
}
