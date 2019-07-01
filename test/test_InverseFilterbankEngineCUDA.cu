#include <complex>
#include <vector>

#include "catch.hpp"

#include "util.hpp"

#include "dsp/InverseFilterbankEngineCUDA.h"


TEST_CASE ("InverseFilterbankEngineCUDA") {}


TEST_CASE ("apodization overlap kernel should produce expected output", "")
{
  unsigned N = 1e6;
  unsigned overlap = 100;
  unsigned N_apod = N - 2*overlap;

  // generate some data.
  std::vector<std::complex<float>> in(N, 2.0);
  std::vector<std::complex<float>> apod(N_apod, 0.0);
  std::vector<std::complex<float>> out(N_apod, 4.0);

  for (unsigned i=0; i<N_apod; i++) {
    apod[i] = std::complex<float>(2.0, 0.0);
  }

  CUDA::InverseFilterbankEngineCUDA::apply_k_apodization_overlap(
    in.data(), apod.data(), out.data(), overlap, N
  );

  // Now we have to check the output array -- every value should be 4.0
  bool allclose = true;
  for (unsigned i=0; i<N_apod; i++) {
    if (out[i] != std::complex<float>(4.0, 4.0)) {
      allclose = false;
    }
  }
  REQUIRE(allclose == true);
}

TEST_CASE ("reponse stitch kernel should produce expected output", "")
{

}

TEST_CASE ("output overlap discard kernel should produce expected output", "")
{

}
