#include "spark_semantic_runtime.h"
#include "spark_mlm_runtime.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
using Weights = std::array<int8_t, 256>;
constexpr uint64_t kHeaderSize = 64;
constexpr uint64_t kMaxBytes = 1ull << 30;
constexpr uint64_t kMaxFeatures = 3'000'000;
uint16_t le16(const uint8_t *p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t le32(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t le64(const uint8_t *p) { return uint64_t(le32(p)) | (uint64_t(le32(p + 4)) << 32); }
uint32_t crc32(const uint8_t *bytes, size_t n) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) {
        crc ^= bytes[i];
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ ((crc & 1u) ? 0xedb88320u : 0u);
    }
    return ~crc;
}
}
struct spark_semantic_encoder {
    uint64_t fingerprint = 0;
    std::array<uint8_t, 32> fallback{};
    std::unordered_map<std::string, Weights> features;
};

extern "C" int32_t spark_semantic_encoder_open_v1(const char *path, spark_semantic_encoder_t **out) {
    if (!path || !out) return SPARK_MLM_EINVAL;
    *out = nullptr;
    try {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return SPARK_MLM_EIO;
        auto length = file.tellg();
        if (length < static_cast<std::streamoff>(kHeaderSize + 32) ||
            length > static_cast<std::streamoff>(kMaxBytes)) return SPARK_MLM_EFORMAT;
        file.seekg(0);
        std::vector<uint8_t> bytes(static_cast<size_t>(length));
        if (!file.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) return SPARK_MLM_EIO;
        const auto *h = bytes.data();
        if (std::string(reinterpret_cast<const char *>(h), 8) != "SPARKENC" ||
            le16(h + 8) != 1 || le16(h + 10) != 0 || le32(h + 12) != 0 ||
            le64(h + 16) > kMaxFeatures || le64(h + 32) != bytes.size() ||
            le64(h + 40) != kHeaderSize || le32(h + 52) != 0 || le64(h + 56) != 0)
            return SPARK_MLM_EFORMAT;
        if (crc32(bytes.data() + kHeaderSize, bytes.size() - kHeaderSize) != le32(h + 48))
            return SPARK_MLM_EFORMAT;
        const uint64_t features = le64(h + 16);
        auto result = std::make_unique<spark_semantic_encoder>();
        result->fingerprint = le64(h + 24);
        std::copy(bytes.begin() + 64, bytes.begin() + 96, result->fallback.begin());
        size_t pos = 96;
        for (uint64_t i = 0; i < features; ++i) {
            if (pos + 2 > bytes.size()) return SPARK_MLM_EFORMAT;
            uint16_t n = le16(bytes.data() + pos);
            pos += 2;
            if (n == 0 || n > 127 || pos + n + 256 > bytes.size()) return SPARK_MLM_EFORMAT;
            std::string key(reinterpret_cast<const char *>(bytes.data() + pos), n);
            for (char c : key) if (c < '!' || c > '~') return SPARK_MLM_EFORMAT;
            pos += n;
            Weights w{};
            for (auto &v : w) {
                int8_t weight = static_cast<int8_t>(bytes[pos++]);
                if (weight < -7 || weight > 7) return SPARK_MLM_EFORMAT;
                v = weight;
            }
            if (!result->features.emplace(std::move(key), w).second) return SPARK_MLM_EFORMAT;
        }
        if (pos != bytes.size()) return SPARK_MLM_EFORMAT;
        *out = result.release();
        return 0;
    } catch (const std::bad_alloc &) { return SPARK_MLM_ENOMEM; }
      catch (...) { return SPARK_MLM_EFORMAT; }
}

extern "C" void spark_semantic_encoder_close_v1(spark_semantic_encoder_t *p) { delete p; }
extern "C" uint64_t spark_semantic_encoder_fingerprint_v1(const spark_semantic_encoder_t *p) {
    return p ? p->fingerprint : 0;
}

extern "C" int32_t spark_semantic_encode_v1(uint64_t encoder_handle, uint64_t token_ids_address,
                                               int32_t token_count, int32_t mask_index,
                                               uint64_t signature_out_address) {
    if (!encoder_handle || !token_ids_address || !signature_out_address || token_count < 1 ||
        token_count > 8192 || mask_index < 0 || mask_index >= token_count) return SPARK_MLM_EINVAL;
    const auto *encoder = reinterpret_cast<const spark_semantic_encoder_t *>(
        static_cast<uintptr_t>(encoder_handle));
    const auto *ids = reinterpret_cast<const uint32_t *>(static_cast<uintptr_t>(token_ids_address));
    auto *out = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(signature_out_address));
    std::array<int32_t, 256> votes{};
    bool matched = false;
    auto vote = [&](const std::string &key, int32_t freq) {
        auto it = encoder->features.find(key);
        if (it == encoder->features.end()) return;
        matched = true;
        for (size_t bit = 0; bit < 256; ++bit) votes[bit] += it->second[bit] * freq;
    };
    try {
        const int left_start = std::max(0, mask_index - 12);
        const int right_end = std::min(token_count, mask_index + 13);
        for (int i = mask_index - 1, distance = 1; i >= left_start; --i, ++distance)
            vote("L" + std::to_string(std::min(distance, 5)) + ":" + std::to_string(ids[i]),
                 distance <= 2 ? 3 : 1);
        for (int i = mask_index + 1, distance = 1; i < right_end; ++i, ++distance)
            vote("R" + std::to_string(std::min(distance, 5)) + ":" + std::to_string(ids[i]),
                 distance <= 2 ? 3 : 1);
        for (int i = left_start; i + 1 < mask_index; ++i)
            vote("LB:" + std::to_string(ids[i]) + ":" + std::to_string(ids[i + 1]), 2);
        for (int i = mask_index + 1; i + 1 < right_end; ++i)
            vote("RB:" + std::to_string(ids[i]) + ":" + std::to_string(ids[i + 1]), 2);
        std::unordered_set<uint32_t> seen;
        for (int i = 0; i < std::min(token_count, 128); ++i)
            if (ids[i] != ids[mask_index] && seen.insert(ids[i]).second)
                vote("B:" + std::to_string(ids[i]), 1);
        if (!matched) {
            std::copy(encoder->fallback.begin(), encoder->fallback.end(), out);
        } else {
            for (int byte = 0; byte < 32; ++byte) {
                uint8_t b = 0;
                for (int bit = 0; bit < 8; ++bit)
                    if (votes[byte * 8 + bit] > 0) b |= uint8_t(1u << bit);
                out[byte] = b;
            }
        }
        return 0;
    } catch (const std::bad_alloc &) { return SPARK_MLM_ENOMEM; }
      catch (...) { return SPARK_MLM_EFORMAT; }
}

extern "C" int32_t spark_semantic_predict_v1(uint64_t encoder_handle, uint64_t index_handle,
                                                uint64_t token_ids_address, int32_t token_count,
                                                int32_t mask_index, int32_t max_candidates,
                                                uint64_t token_out_address, uint64_t score_out_address) {
    if (!encoder_handle || !index_handle) return SPARK_MLM_EINVAL;
    auto *encoder = reinterpret_cast<const spark_semantic_encoder_t *>(static_cast<uintptr_t>(encoder_handle));
    auto *index = reinterpret_cast<const spark_mlm_index_t *>(static_cast<uintptr_t>(index_handle));
    if (encoder->fingerprint == 0 || encoder->fingerprint != spark_mlm_index_fingerprint_v1(index))
        return SPARK_MLM_EFORMAT;
    std::array<uint8_t, 32> sig{};
    const int32_t status = spark_semantic_encode_v1(encoder_handle, token_ids_address, token_count,
                                                    mask_index, reinterpret_cast<uint64_t>(sig.data()));
    if (status < 0) return status;
    return spark_mlm_probe_v1(index_handle, reinterpret_cast<uint64_t>(sig.data()),
                             max_candidates, token_out_address, score_out_address);
}
