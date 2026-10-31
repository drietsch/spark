#ifndef SPARK_MLM_RUNTIME_H
#define SPARK_MLM_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * SPARK MLM probe ABI v1 (LP64 / 64-bit little-endian hosts).
 * Addresses are uint64_t intentionally: the MLIR v0.2 op uses only builtin
 * integer types and its lowering creates an ordinary LLVM function call.
 * Never accept untrusted addresses from a remote process.
 *
 * Index is immutable and shareable between threads after open.
 * Returns 0 on successful open, negative status on error.
 */
#define SPARK_MLM_ABI_VERSION 1
#define SPARK_MLM_MAX_CANDIDATES 64
#define SPARK_MLM_ENCODED_BYTES 32

typedef struct spark_mlm_index spark_mlm_index_t;

enum {
    SPARK_MLM_EINVAL = -1,
    SPARK_MLM_EFORMAT = -2,
    SPARK_MLM_EIO = -3,
    SPARK_MLM_ENOMEM = -4,
    SPARK_MLM_EEMPTY = -5
};

int32_t spark_mlm_index_open_v1(const char *path, spark_mlm_index_t **out);
void spark_mlm_index_close_v1(spark_mlm_index_t *index);
/* 0 for baseline; nonzero semantic index binding fingerprint. */
uint64_t spark_mlm_index_fingerprint_v1(const spark_mlm_index_t *index);

/*
 * Arguments: index handle, ptr to 32 signature bytes, top-k (1..64),
 * ptr to 64 int32 token slots, ptr to 64 int32 signed Q16 score slots.
 * Returns number of active candidates (0..64) or a negative error.
 * Inactive token slots are -1 and scores are zero. All data little-endian.
 * Candidate scores aggregate up to eight nearest LSH-retrieved entries.
 * Retrieval is limited to entries matching at least one configured band;
 * no full-index fallback. The distance weight is
 *      ((512 - hamming_distance) / 256)^4.
 * Probability is quantized to Q16 before multiplication, then rounded.
 * This is an *integer* approximation of the Python reference scoring.
 *
 * The open function must succeed before invoking this. The caller must keep
 * the index alive until all concurrent calls finish. Buffer regions must not
 * overlap with each other or with the index's internal storage.
 */
int32_t spark_mlm_probe_v1(uint64_t index_handle,
                           uint64_t signature_address,
                           int32_t max_candidates,
                           uint64_t token_out_address,
                           uint64_t score_out_address);

#ifdef __cplusplus
}
#endif
#endif
