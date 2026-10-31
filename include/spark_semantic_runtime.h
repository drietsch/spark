#ifndef SPARK_SEMANTIC_RUNTIME_H
#define SPARK_SEMANTIC_RUNTIME_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* x86-64 little-endian, trusted local-process pointers, C ABI v1. */
typedef struct spark_semantic_encoder spark_semantic_encoder_t;

/* Returns 0 on success. Negative values use SPARK_MLM_* error conventions. */
int32_t spark_semantic_encoder_open_v1(const char *path, spark_semantic_encoder_t **out);
void spark_semantic_encoder_close_v1(spark_semantic_encoder_t *encoder);
uint64_t spark_semantic_encoder_fingerprint_v1(const spark_semantic_encoder_t *encoder);

/* Arguments: opaque encoder handle, pointer to uint32_t tokens, token count (1..8192),
 * mask index, pointer to 32 output bytes. 0 on success; negative on error.
 * The output is little-endian and exactly matches SemanticEncoder.encode() in Python.
 * Buffers must be valid, non-overlapping, and owned by the caller.
 */
int32_t spark_semantic_encode_v1(uint64_t encoder_handle,
                                uint64_t token_ids_address,
                                int32_t token_count,
                                int32_t mask_index,
                                uint64_t signature_out_address);

/* Full native path. Validates index <-> encoder fingerprint before encoding.
 * Calls the regular spark_mlm_probe_v1 after encoding and returns candidate count.
 * Out-of-range/error: negative status. Output arrays each have 64 int32 elements.
 */
int32_t spark_semantic_predict_v1(uint64_t encoder_handle,
                                 uint64_t index_handle,
                                 uint64_t token_ids_address,
                                 int32_t token_count,
                                 int32_t mask_index,
                                 int32_t max_candidates,
                                 uint64_t token_out_address,
                                 uint64_t score_out_address);
#ifdef __cplusplus
}
#endif
#endif
