#include "catch.hpp"

#include "dsp/FilterbankEngineCUDA.h"


TEST_CASE ("Can create instance of FilterbankEngineCUDA")
{
  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::FilterbankEngine engine(cuda_stream);
}
