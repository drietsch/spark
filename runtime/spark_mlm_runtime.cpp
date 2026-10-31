#include "spark_mlm_runtime.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct Candidate { uint32_t token; uint32_t prob_q16; };
struct Entry { std::array<uint8_t, 32> sig; std::vector<Candidate> candidates; };
struct BandKey {
    uint32_t band;
    uint32_t value;
    bool operator==(const BandKey &r) const { return band == r.band && value == r.value; }
};
struct BandHash {
    size_t operator()(const BandKey &k) const {
        return (static_cast<size_t>(k.band) << 32) ^ static_cast<size_t>(k.value);
    }
};
constexpr uint64_t kHeader = 64;
constexpr uint64_t kRecBytes = 48;
constexpr uint64_t kCandBytes = 8;
constexpr uint64_t kMaxRecords = 10'000'000;
constexpr uint64_t kMaxCandidates = 100'000'000;

uint16_t u16(const uint8_t *p) {
    return static_cast<uint16_t>(p[0] | (uint16_t(p[1]) << 8));
}
uint32_t u32(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t u64(const uint8_t *p) {
    return uint64_t(u32(p)) | (uint64_t(u32(p + 4)) << 32);
}
uint32_t popcount(uint8_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(v)));
#else
    uint32_t bits = 0;
    for (; v; v &= static_cast<uint8_t>(v - 1)) ++bits;
    return bits;
#endif
}
uint32_t hamming(const std::array<uint8_t, 32> &a, const uint8_t *b) {
    uint32_t d = 0;
    for (size_t i = 0; i < 32; ++i) d += popcount(static_cast<uint8_t>(a[i] ^ b[i]));
    return d;
}
uint32_t band_value(const uint8_t *sig, uint32_t band, uint32_t bits) {
    if (bits == 32) return u32(sig + band * 4);
    return sig[band];
}
uint32_t prob_q16(float value) {
    return static_cast<uint32_t>(std::llround(static_cast<double>(value) * 65536.0));
}
uint64_t contribution(uint32_t prob, uint32_t distance) {
    const uint64_t a = 512u - distance;
    const uint64_t w4 = a * a * a * a;
    const uint64_t denominator = uint64_t(1) << 32;
    return (uint64_t(prob) * w4 + denominator / 2) / denominator;
}
} // namespace

struct spark_mlm_index {
    std::vector<Entry> entries;
    std::unordered_map<BandKey, std::vector<uint32_t>, BandHash> buckets;
    uint32_t band_bits;
    uint32_t bands;
    uint64_t encoder_fingerprint;
};

extern "C" int32_t spark_mlm_index_open_v1(const char *path, spark_mlm_index_t **out) {
    if (!path || !out) return SPARK_MLM_EINVAL;
    *out = nullptr;
    try {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return SPARK_MLM_EIO;
        const auto file_size = f.tellg();
        if (file_size < static_cast<std::streamoff>(kHeader)) return SPARK_MLM_EFORMAT;
        f.seekg(0);
        std::array<uint8_t, kHeader> hdr{};
        if (!f.read(reinterpret_cast<char *>(hdr.data()), hdr.size())) return SPARK_MLM_EIO;
        if (std::memcmp(hdr.data(), "SPARKMLM", 8) != 0 || u16(hdr.data() + 8) != 1 ||
            u16(hdr.data() + 10) != 0) return SPARK_MLM_EFORMAT;
        const uint32_t flags = u32(hdr.data() + 12);
        const uint64_t n = u64(hdr.data() + 16), nc = u64(hdr.data() + 24);
        const uint64_t rec_off = u64(hdr.data() + 32), data_off = u64(hdr.data() + 40);
        const uint64_t total = u64(hdr.data() + 48), fingerprint = u64(hdr.data() + 56);
        if (flags > 1 || n > kMaxRecords || nc > kMaxCandidates ||
            (flags == 0 && fingerprint != 0) || rec_off != kHeader ||
            data_off != kHeader + n * kRecBytes || total != data_off + nc * kCandBytes ||
            total != static_cast<uint64_t>(file_size)) return SPARK_MLM_EFORMAT;
        auto index = std::make_unique<spark_mlm_index>();
        index->encoder_fingerprint = fingerprint;
        index->band_bits = flags ? 8 : 32;
        index->bands = 256 / index->band_bits;
        std::vector<uint8_t> records(static_cast<size_t>(n * kRecBytes));
        std::vector<uint8_t> candidate_bytes(static_cast<size_t>(nc * kCandBytes));
        if ((!records.empty() && !f.read(reinterpret_cast<char *>(records.data()), records.size())) ||
            (!candidate_bytes.empty() && !f.read(reinterpret_cast<char *>(candidate_bytes.data()), candidate_bytes.size())))
            return SPARK_MLM_EIO;
        index->entries.reserve(static_cast<size_t>(n));
        for (uint64_t i = 0; i < n; ++i) {
            const auto *rec = records.data() + i * kRecBytes;
            const auto offset = u64(rec + 32);
            const auto count = u32(rec + 40);
            if (u32(rec + 44) != 0 || count == 0 || offset > nc || count > nc - offset)
                return SPARK_MLM_EFORMAT;
            Entry e{};
            std::copy(rec, rec + 32, e.sig.begin());
            e.candidates.reserve(count);
            for (uint32_t j = 0; j < count; ++j) {
                const auto *candidate = candidate_bytes.data() + (offset + j) * kCandBytes;
                uint32_t raw = u32(candidate + 4);
                float p;
                std::memcpy(&p, &raw, sizeof(p));
                if (!std::isfinite(p) || p < 0 || p > 1) return SPARK_MLM_EFORMAT;
                e.candidates.push_back({u32(candidate), prob_q16(p)});
            }
            index->entries.push_back(std::move(e));
        }
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t band = 0; band < index->bands; ++band) {
                index->buckets[{band, band_value(index->entries[i].sig.data(), band, index->band_bits)}].push_back(i);
            }
        }
        *out = index.release();
        return 0;
    } catch (const std::bad_alloc &) { return SPARK_MLM_ENOMEM; }
      catch (...) { return SPARK_MLM_EFORMAT; }
}

extern "C" void spark_mlm_index_close_v1(spark_mlm_index_t *index) { delete index; }

extern "C" uint64_t spark_mlm_index_fingerprint_v1(const spark_mlm_index_t *index) {
    return index ? index->encoder_fingerprint : 0;
}

extern "C" int32_t spark_mlm_probe_v1(uint64_t index_handle, uint64_t signature_address,
                                         int32_t max_candidates, uint64_t token_out_address,
                                         uint64_t score_out_address) {
    if (!index_handle || !signature_address || !token_out_address || !score_out_address ||
        max_candidates < 1 || max_candidates > SPARK_MLM_MAX_CANDIDATES) return SPARK_MLM_EINVAL;
    auto *index = reinterpret_cast<const spark_mlm_index_t *>(static_cast<uintptr_t>(index_handle));
    const auto *sig = reinterpret_cast<const uint8_t *>(static_cast<uintptr_t>(signature_address));
    auto *tokens = reinterpret_cast<int32_t *>(static_cast<uintptr_t>(token_out_address));
    auto *scores = reinterpret_cast<int32_t *>(static_cast<uintptr_t>(score_out_address));
    for (int i = 0; i < SPARK_MLM_MAX_CANDIDATES; ++i) { tokens[i] = -1; scores[i] = 0; }
    try {
        std::vector<uint32_t> candidates;
        for (uint32_t band = 0; band < index->bands; ++band) {
            const auto it = index->buckets.find({band, band_value(sig, band, index->band_bits)});
            if (it != index->buckets.end()) candidates.insert(candidates.end(), it->second.begin(), it->second.end());
        }
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        if (candidates.empty()) return 0;
        std::vector<std::pair<uint32_t, uint32_t>> ranked;
        ranked.reserve(candidates.size());
        for (uint32_t id : candidates) ranked.emplace_back(hamming(index->entries[id].sig, sig), id);
        const size_t count = std::min<size_t>(8, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + count, ranked.end());
        std::map<uint32_t, uint64_t> votes;
        for (size_t i = 0; i < count; ++i) {
            const auto &entry = index->entries[ranked[i].second];
            for (const auto &candidate : entry.candidates) {
                votes[candidate.token] += contribution(candidate.prob_q16, ranked[i].first);
            }
        }
        std::vector<std::pair<uint32_t, uint64_t>> ordered(votes.begin(), votes.end());
        std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
            return a.second != b.second ? a.second > b.second : a.first < b.first;
        });
        const size_t active = std::min<size_t>(ordered.size(), static_cast<size_t>(max_candidates));
        for (size_t i = 0; i < active; ++i) {
            // The sentinel -1 requires token IDs to be <= INT32_MAX.
            if (ordered[i].first > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
                return SPARK_MLM_EFORMAT;
            tokens[i] = static_cast<int32_t>(ordered[i].first);
            scores[i] = static_cast<int32_t>(std::min<uint64_t>(ordered[i].second, INT32_MAX));
        }
        return static_cast<int32_t>(active);
    } catch (const std::bad_alloc &) { return SPARK_MLM_ENOMEM; }
      catch (...) { return SPARK_MLM_EFORMAT; }
}
