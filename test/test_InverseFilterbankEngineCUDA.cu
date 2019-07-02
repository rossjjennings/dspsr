#include <complex>
#include <vector>
#include <iostream>

#include "catch.hpp"

#include "util.hpp"

#include "dsp/InverseFilterbankEngineCUDA.h"


TEST_CASE ("InverseFilterbankEngineCUDA") {}


TEST_CASE ("apodization overlap kernel should produce expected output", "")
{
  int N = 1e4;
  int nchan = 256;
  int overlap = 100;
  int N_apod = N - 2*overlap;

  // generate some data.
  std::vector<std::complex<float>> in(N*nchan, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> apod(N_apod, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out(N_apod*nchan, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  for (int i=0; i<nchan; i++) {
    for (int j=0; j<N; j++) {
      in[i*N + j] = std::complex<float>(i+1, i+1);
    }
  }

  // apodization filter is just multiplying by 2.
  for (int i=0; i<N_apod; i++) {
    apod[i] = std::complex<float>(2.0, 0.0);
  }


  CUDA::InverseFilterbankEngineCUDA::apply_k_apodization_overlap(
    in.data(), apod.data(), out.data(), overlap, N, nchan
  );

  // Now we have to check the output array -- every value should be double initial value
  bool allclose = true;
  std::complex<float> comp;
  for (int i=0; i<nchan; i++) {
    comp = std::complex<float>(2*(i+1), 2*(i+1));
    for (int j=0; j<N_apod; j++) {
      // std::cerr << "[" << i << "," << j << "] " << out[i*N_apod + j] << std::endl;
      if (out[i*N_apod + j] != comp) {
        allclose = false;
      }
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
