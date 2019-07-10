#include <complex>
#include <vector>
#include <iostream>

#include "catch.hpp"

#include "util.hpp"
#include "util.cuda.hpp"

#include "Rational.h"
#include "dsp/InverseFilterbankEngineCUDA.h"

const float thresh = 1e-5;

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
    in, apod, out, overlap, N, nchan
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

  int npart = 10;
  int ndat = 8;
  int nchan = 4;
  int npol = 2;

  Rational os_factor(4, 3);
  int in_ndat_keep = os_factor.normalize(ndat);
  int out_ndat = nchan * in_ndat_keep;
  int out_size = out_ndat * npol * npart;
  int in_size = ndat * nchan * npol * npart;
  // generate some data.
  std::vector<std::complex<float>> in(in_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> resp(out_ndat, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  int in_idx;
  float val;
  for (int ipart=0; ipart<npart; ipart++) {
    for (int ipol=0; ipol<npol; ipol++) {
      val = 0;
      for (int ichan=0; ichan<nchan; ichan++) {
        for (int idat=0; idat<ndat; idat++) {
          val += 1;
          in_idx = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat + idat;
          in[in_idx] = std::complex<float>(val, val);
        }
      }
    }
  }

  std::vector<int> dim_in = {npart, npol, nchan, ndat};
  std::vector<int> dim_out = {npart, npol, out_ndat};

  // util::print_array<std::complex<float>>(in, dim_in);

  // response is just multiplying by 2.
  for (int i=0; i<out_ndat; i++) {
    resp[i] = std::complex<float>(2.0, 0.0);
  }

  std::vector<bool> pfb_dc_chan = {true, false};
  std::vector<bool> pfb_all_chan = {true, false};

  // std::vector<bool> pfb_dc_chan = {true}; //, false};
  // std::vector<bool> pfb_all_chan = {true}; //, false};
  bool allclose = true;
  for (auto dc_it=pfb_dc_chan.begin(); dc_it != pfb_dc_chan.end(); dc_it++) {
    for (auto all_it=pfb_all_chan.begin(); all_it != pfb_all_chan.end(); all_it++){
      out_cpu.assign(out_cpu.size(), std::complex<float>(0.0, 0.0));
      out_gpu.assign(out_gpu.size(), std::complex<float>(0.0, 0.0));

      util::response_stitch_cpu<std::complex<float>>(
        in, resp, out_cpu, os_factor, npart, npol, nchan, ndat, *dc_it, *all_it
      );

      CUDA::InverseFilterbankEngineCUDA::apply_k_response_stitch(
        in, resp, out_gpu, os_factor, npart, npol, nchan, ndat, *dc_it, *all_it
      );

      // util::print_array<std::complex<float>>(out_cpu, dim_out);
      // util::print_array<std::complex<float>>(out_gpu, dim_out);
      allclose = util::allclose<std::complex<float>>(out_cpu, out_gpu, thresh);

      REQUIRE(allclose == true);
    }
  }
}

TEST_CASE ("output overlap discard kernel should produce expected output", "")
{
  int ndat = 10;
  int nchan = 3;
  int npol = 2;
  int discard = 2;

  dim3 in_dim(npol, nchan, ndat);
  dim3 out_dim(npol, nchan, ndat - 2*discard);

  // generate some data.
  std::vector<std::complex<float>> in(
    in_dim.x * in_dim.y * in_dim.z, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out(
    out_dim.x * out_dim.y * out_dim.z, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  int in_idx;
  float val;
  for (int ipol=0; ipol<npol; ipol++) {
    for (int ichan=0; ichan<nchan; ichan++) {
      val = (ipol + 1) * (ichan + 1);
      for (int idat=0; idat<ndat; idat++) {
        in_idx = ipol*nchan*ndat + ichan*ndat + idat;
        in[in_idx] = std::complex<float>(val, val);
      }
    }
  }

  // util::print_array(in, in_dim);

  CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard(
    in, in_dim, out, out_dim, discard
  );

  // Now we have to check the output array -- every value should be double initial value
  bool allclose = true;
  std::complex<float> comp;
  int out_idx;
  for (int ipol=0; ipol<npol; ipol++) {
    for (int ichan=0; ichan<nchan; ichan++) {
      val = (ipol + 1) * (ichan + 1);
      comp = std::complex<float>(val, val);
      for (int idat=0; idat < out_dim.z; idat++) {
        out_idx = ipol*nchan*out_dim.z + ichan*out_dim.z + idat;
        if (out[out_idx] != comp) {
          allclose = false;
        }
      }
    }
  }

  // util::print_array(out, out_dim);

  REQUIRE(allclose == true);
}
