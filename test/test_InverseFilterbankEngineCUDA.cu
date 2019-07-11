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



TEST_CASE ("output overlap discard kernel should produce expected output", "")
{
  int npart = 3;
  int npol = 1;
  int nchan = 4;
  int ndat = 4;
  int overlap = 1;

  int out_ndat = ndat - 2*overlap;

  int in_size = npart * npol * nchan * ndat;
  int out_size = npart * npol * nchan * out_ndat;

  // generate some data.
  std::vector<std::complex<float>> in(in_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  int idx;
  float val;

  for (int ipart=0; ipart<npart; ipart++) {
    for (int ipol=0; ipol<npol; ipol++) {
      val = 0;
      for (int ichan=0; ichan<nchan; ichan++) {
        for (int idat=0; idat<ndat; idat++) {
          idx = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat + idat;
          in[idx] = std::complex<float>(val, val);
          val++;
        }
      }
    }
  }
  std::vector<int> in_dim = {npart, npol, nchan, ndat};
  std::vector<int> out_dim = {npart, npol, nchan, out_ndat};

  util::overlap_discard_cpu(
    in, out_cpu, overlap, npart, npol, nchan, ndat
  );

  CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard(
    in, out_gpu, overlap, npart, npol, nchan, ndat
  );

  bool allclose = util::allclose(out_cpu, out_gpu, thresh);
  REQUIRE(allclose == true);
}



TEST_CASE ("apodization overlap kernel should produce expected output", "")
{
  int npart = 3;
  int npol = 1;
  int nchan = 4;
  int ndat = 4;
  int overlap = 1;

  int out_ndat = ndat - 2*overlap;

  int in_size = npart * npol * nchan * ndat;
  int out_size = npart * npol * nchan * out_ndat;
  int apod_size = nchan * out_ndat;

  // generate some data.
  std::vector<std::complex<float>> in(in_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> apod(apod_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  int idx;
  float val;

  for (int ipart=0; ipart<npart; ipart++) {
    for (int ipol=0; ipol<npol; ipol++) {
      val = 0;
      for (int ichan=0; ichan<nchan; ichan++) {
        for (int idat=0; idat<ndat; idat++) {
          idx = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat + idat;
          in[idx] = std::complex<float>(val, val);
          val++;
        }
      }
    }
  }
  std::vector<int> in_dim = {npart, npol, nchan, ndat};
  std::vector<int> out_dim = {npart, npol, nchan, out_ndat};

  // apodization filter is just multiplying by 2.
  for (int i=0; i<apod_size; i++) {
    apod[i] = std::complex<float>(2.0, 0.0);
  }

  util::apodization_overlap_cpu< std::complex<float> >(
    in, apod, out_cpu, overlap, npart, npol, nchan, ndat
  );

  CUDA::InverseFilterbankEngineCUDA::apply_k_apodization_overlap(
    in, apod, out_gpu, overlap, npart, npol, nchan, ndat
  );

  bool allclose = util::allclose(out_cpu, out_gpu, thresh);
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
