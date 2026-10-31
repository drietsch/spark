#include "spark_runtime.h"
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: spark_demo model.idx prompt_token [prompt_token...]\n";
    return 2;
  }
  spark_model_t *model = nullptr;
  auto rc = spark_model_open(argv[1], &model);
  if (rc != SPARK_OK) { std::cerr << "model open failed: " << rc << "\n"; return 1; }
  spark_session_t *session = nullptr;
  rc = spark_session_open(model, &session);
  if (rc != SPARK_OK) { spark_model_close(model); return 1; }
  for (int i = 2; i < argc; ++i) {
    char *end = nullptr;
    auto value = std::strtoull(argv[i], &end, 10);
    if (!end || *end || value > UINT32_MAX) { std::cerr << "bad token\n"; return 2; }
    spark_session_feed(session, static_cast<uint32_t>(value));
  }
  uint32_t next = 0;
  spark_step_stats_t stats{};
  stats.abi_size = sizeof(stats);
  rc = spark_session_step(session, &next, &stats);
  if (rc == SPARK_OK) std::cout << "next=" << next << " order=" << stats.matched_order
                                << " candidates=" << stats.matched_candidates << "\n";
  else std::cout << "step_status=" << rc << "\n";
  spark_session_close(session);
  spark_model_close(model);
  return rc == SPARK_OK ? 0 : 1;
}
