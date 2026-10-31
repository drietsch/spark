#include "spark_runtime.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <vector>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "SPARK v0.1 reference loader supports only little-endian targets"
#endif

#pragma pack(push, 1)
struct DiskHeader {
  char magic[8];
  uint16_t major, minor;
  uint32_t flags;
  uint64_t bucket_count, entry_count, seed, bucket_offset, entry_offset, file_bytes;
};
struct DiskEntry {
  uint64_t key;
  uint32_t token;
  uint32_t count;
};
#pragma pack(pop)
static_assert(sizeof(DiskHeader) == 64, "SPARK header must be 64 bytes");
static_assert(sizeof(DiskEntry) == 16, "SPARK record must be 16 bytes");

struct spark_model { std::vector<DiskEntry> rows; };
struct spark_session {
  const spark_model *model;
  std::array<uint32_t, 4> context{};
  unsigned used = 0;
};

static uint64_t signature(const uint32_t *tokens, unsigned len) {
  uint64_t h = 14695981039346656037ULL; // FNV-1a, 64-bit
  for (unsigned i = 0; i < len; ++i) {
    for (unsigned j = 0; j < 4; ++j) {
      h ^= static_cast<uint8_t>(tokens[i] >> (8 * j));
      h *= 1099511628211ULL;
    }
  }
  return h;
}

extern "C" {
spark_status_t spark_model_open(const char *path, spark_model_t **out) {
  if (!path || !out) return SPARK_EINVAL;
  *out = nullptr;
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return SPARK_EIO;
  const auto end = f.tellg();
  if (end < static_cast<std::streamoff>(sizeof(DiskHeader))) return SPARK_EFORMAT;
  const uint64_t bytes = static_cast<uint64_t>(end);
  f.seekg(0);
  DiskHeader h{};
  f.read(reinterpret_cast<char *>(&h), sizeof(h));
  if (!f || std::memcmp(h.magic, "SPARKIDX", 8) != 0 || h.major != 1 ||
      h.minor != 0 || h.flags != 0 || h.seed != 0 || h.bucket_count != 0 ||
      h.bucket_offset != 0 || h.entry_offset != sizeof(DiskHeader) ||
      h.file_bytes != bytes || h.entry_count > (bytes - sizeof(DiskHeader)) / sizeof(DiskEntry) ||
      h.entry_count != (bytes - sizeof(DiskHeader)) / sizeof(DiskEntry)) return SPARK_EFORMAT;
  // Safety guard: this simple MVP uses a vector, not mmap. Production must use mmap.
  if (h.entry_count > 100000000ULL) return SPARK_EUNSUPPORTED;
  try {
    auto *m = new spark_model();
    m->rows.resize(static_cast<size_t>(h.entry_count));
    if (!m->rows.empty()) f.read(reinterpret_cast<char *>(m->rows.data()),
                                   static_cast<std::streamsize>(m->rows.size() * sizeof(DiskEntry)));
    const auto less = [](const DiskEntry &a, const DiskEntry &b) {
      return a.key < b.key || (a.key == b.key && a.token < b.token);
    };
    const auto duplicate = [](const DiskEntry &a, const DiskEntry &b) {
      return a.key == b.key && a.token == b.token;
    };
    if (!f || !std::is_sorted(m->rows.begin(), m->rows.end(), less) ||
        std::adjacent_find(m->rows.begin(), m->rows.end(), duplicate) != m->rows.end()) {
      delete m; return SPARK_EFORMAT;
    }
    *out = m;
    return SPARK_OK;
  } catch (const std::bad_alloc &) { return SPARK_ENOMEM; }
}

void spark_model_close(spark_model_t *m) { delete m; }

spark_status_t spark_session_open(const spark_model_t *m, spark_session_t **out) {
  if (!m || !out) return SPARK_EINVAL;
  *out = nullptr;
  auto *s = new (std::nothrow) spark_session();
  if (!s) return SPARK_ENOMEM;
  s->model = m; *out = s; return SPARK_OK;
}
void spark_session_close(spark_session_t *s) { delete s; }
spark_status_t spark_session_reset(spark_session_t *s) {
  if (!s) return SPARK_EINVAL;
  s->used = 0; return SPARK_OK;
}
spark_status_t spark_session_feed(spark_session_t *s, uint32_t token) {
  if (!s) return SPARK_EINVAL;
  if (s->used < s->context.size()) s->context[s->used++] = token;
  else {
    std::move(s->context.begin() + 1, s->context.end(), s->context.begin());
    s->context.back() = token;
  }
  return SPARK_OK;
}
spark_status_t spark_session_step(spark_session_t *s, uint32_t *out,
                                 spark_step_stats_t *stats) {
  if (!s || !out) return SPARK_EINVAL;
  if (stats && stats->abi_size != sizeof(spark_step_stats_t)) return SPARK_EINVAL;
  if (stats) { stats->matched_order = 0; stats->matched_candidates = 0; }
  const auto &rows = s->model->rows;
  for (unsigned n = s->used; n >= 1; --n) {
    const auto key = signature(s->context.data() + s->used - n, n);
    auto it = std::lower_bound(rows.begin(), rows.end(), key,
         [](const DiskEntry &e, uint64_t k) { return e.key < k; });
    if (it == rows.end() || it->key != key) continue;
    uint32_t best_token = it->token, best_count = 0;
    uint64_t matched = 0;
    for (; it != rows.end() && it->key == key; ++it) {
      ++matched;
      if (it->count > best_count || (it->count == best_count && it->token < best_token)) {
        best_token = it->token; best_count = it->count;
      }
    }
    *out = best_token;
    if (stats) { stats->matched_order = n; stats->matched_candidates = matched; }
    return spark_session_feed(s, *out);
  }
  return SPARK_ENOMATCH;
}
} // extern "C"
