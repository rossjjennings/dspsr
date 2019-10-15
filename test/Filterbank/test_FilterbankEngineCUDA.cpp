#include "catch.hpp"

#include "dsp/FilterbankEngineCUDA.h"

#include "util/util.hpp"

TEST_CASE ("Can create instance of FilterbankEngineCUDA", "[unit][cuda][no_file][FilterbankEngineCUDA]")
{
  if (test::util::config::verbose) {
    std::cerr << "test_FilterbankEngineCUDA" << std::endl;
  }
  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::FilterbankEngine engine(cuda_stream);
}
