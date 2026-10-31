#ifndef SPARK_RUNTIME_H
#define SPARK_RUNTIME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_ABI_MAJOR 1u
#define SPARK_ABI_MINOR 0u

typedef struct spark_model spark_model_t;
typedef struct spark_session spark_session_t;

typedef enum spark_status {
  SPARK_OK = 0,
  SPARK_EINVAL = 1,
  SPARK_EIO = 2,
  SPARK_EFORMAT = 3,
  SPARK_ENOMEM = 4,
  SPARK_ENOMATCH = 5,
  SPARK_EUNSUPPORTED = 6
} spark_status_t;

typedef struct spark_step_stats {
  uint32_t abi_size;   /* caller sets sizeof(spark_step_stats_t) */
  uint32_t matched_order;
  uint64_t matched_candidates;
  uint64_t reserved[2];
} spark_step_stats_t;

spark_status_t spark_model_open(const char *index_path, spark_model_t **out_model);
void spark_model_close(spark_model_t *model);
spark_status_t spark_session_open(const spark_model_t *model, spark_session_t **out_session);
void spark_session_close(spark_session_t *session);
spark_status_t spark_session_reset(spark_session_t *session);
spark_status_t spark_session_feed(spark_session_t *session, uint32_t token_id);
spark_status_t spark_session_step(spark_session_t *session, uint32_t *next_token,
                                  spark_step_stats_t *optional_stats);

#ifdef __cplusplus
}
#endif
#endif
