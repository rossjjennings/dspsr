#include <complex>
#include <vector>
#include <iostream>

#include "catch.hpp"

#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/InverseFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "Rational.h"

#include "util/util.hpp"
#include "InverseFilterbankTestConfig.hpp"

void check_error (const char*);

static test::util::InverseFilterbank::InverseFilterbankTestConfig test_config;

TEST_CASE (
  "output overlap discard kernel should produce expected output",
  "[no_file][cuda][overlap_discard]"
)
{
  std::vector<float> thresh = test_config.get_thresh();
  std::vector<test::util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  auto idx = GENERATE_COPY(range(0, (int) test_shapes.size()));

	test::util::TestShape test_shape = test_shapes[idx];

  unsigned npart = test_shape.npart;
  unsigned npol = test_shape.input_npol;
  unsigned nchan = test_shape.input_nchan;
  unsigned ndat = test_shape.input_ndat;
  unsigned overlap = test_shape.overlap_pos;

  unsigned total_discard = 2*overlap;
  unsigned step = ndat - total_discard;

  unsigned in_total_ndat = npart * step + total_discard;
  unsigned out_total_ndat = npart * ndat;

  unsigned in_size = npol * nchan * in_total_ndat;
  unsigned out_size = npol * nchan * out_total_ndat;

  // generate some data.
  std::vector<std::complex<float>> in(in_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  unsigned in_idx;
  float val;

  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      val = ichan*npol + ipol;
      for (unsigned ipart=0; ipart<npart; ipart++) {
        for (unsigned idat=0; idat<step; idat++) {
          in_idx = ichan*npol*in_total_ndat + ipol*in_total_ndat + ipart*step + idat;
          in[in_idx] = std::complex<float>(val, val);
          val++;
        }
      }
    }
  }
  std::vector<unsigned> in_dim = {nchan, npol, in_total_ndat};
  std::vector<unsigned> out_dim = {nchan, npol, out_total_ndat};

  auto t = test::util::now();
  test::util::InverseFilterbank::overlap_discard_cpu_FPT(
    in, out_cpu, overlap, npart, npol, nchan, ndat, in_total_ndat, out_total_ndat
  );
  test::util::delta<std::milli> delta_cpu = test::util::now() - t;

  t = test::util::now();
  CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard(
    in, out_gpu, overlap, npart, npol, nchan, ndat
  );

  // test::util::print_array<std::complex<float>>(in, in_dim);
  // test::util::print_array<std::complex<float>>(out_cpu, out_dim);
  // test::util::print_array<std::complex<float>>(out_gpu, out_dim);

  test::util::delta<std::milli> delta_gpu = test::util::now() - t;
  if (test::util::config::verbose) {
    std::cerr << "overlap discard GPU: " << delta_gpu.count()
      << " ms; CPU: " << delta_cpu.count()
      << " ms; CPU/GPU: " << delta_cpu.count() / delta_gpu.count()
      << std::endl;
  }

  bool allclose = test::util::allclose(out_cpu, out_gpu, thresh[0]);
  REQUIRE(allclose == true);
}



TEST_CASE (
  "overlap save kernel should produce expected output",
  "[no_file][cuda][overlap_save]"
)
{
  std::vector<float> thresh = test_config.get_thresh();
  std::vector<test::util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  auto idx = GENERATE_COPY(range(0, (int) test_shapes.size()));

  test::util::TestShape test_shape = test_shapes[idx];

  unsigned npart = test_shape.npart;
  unsigned npol = test_shape.input_npol;
  unsigned nchan = test_shape.input_nchan;
  unsigned ndat = test_shape.input_ndat;
  unsigned overlap = test_shape.overlap_pos;

  unsigned total_discard = 2*overlap;
  unsigned step = ndat - total_discard;

  unsigned in_total_ndat = npart * ndat;
  unsigned out_total_ndat = npart * step;

  unsigned in_size = npol * nchan * in_total_ndat;
  unsigned out_size = npol * nchan * out_total_ndat;

  // generate some data.
  std::vector< std::complex<float> > in(in_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  unsigned in_idx;
  float val;

  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      val = ichan*npol + ipol;
      for (unsigned ipart=0; ipart<npart; ipart++) {
        for (unsigned idat=0; idat<ndat; idat++) {
          in_idx = ichan*npol*in_total_ndat + ipol*in_total_ndat + ipart*ndat + idat;
          in[in_idx] = std::complex<float>(val, val);
          val++;
        }
      }
    }
  }


  std::vector<unsigned> in_dim = {nchan, npol, in_total_ndat};
  std::vector<unsigned> out_dim = {nchan, npol, out_total_ndat};

  // test::util::print_array(in, in_dim);

  auto t = test::util::now();
  test::util::InverseFilterbank::overlap_save_cpu_FPT< std::complex<float> >(
    in, out_cpu, overlap, npart, npol, nchan, ndat, in_total_ndat, out_total_ndat
  );

  // test::util::print_array(out_cpu, out_dim);

  test::util::delta<std::milli> delta_cpu = test::util::now() - t;

  t = test::util::now();
  CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_save(
    in, out_gpu, overlap, npart, npol, nchan, ndat
  );
  test::util::delta<std::milli> delta_gpu = test::util::now() - t;

  if (test::util::config::verbose) {
    std::cerr << "overlap save GPU: " << delta_gpu.count()
      << " ms; CPU: " << delta_cpu.count()
      << " ms; CPU/GPU: " << delta_cpu.count() / delta_gpu.count()
      << std::endl;
  }

  bool allclose = test::util::allclose(out_cpu, out_gpu, thresh[0]);
  REQUIRE(allclose == true);
}



TEST_CASE (
  "apodization overlap kernel should produce expected output",
  "[no_file][cuda][apodization_overlap]"
)
{
  std::vector<float> thresh = test_config.get_thresh();
  std::vector<test::util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  auto idx = GENERATE_COPY(range(0, (int) test_shapes.size()));

	test::util::TestShape test_shape = test_shapes[idx];

  unsigned npart = test_shape.npart;
  unsigned npol = test_shape.input_npol;
  unsigned nchan = test_shape.input_nchan;
  unsigned ndat = test_shape.input_ndat;
  unsigned overlap = test_shape.overlap_pos;

  unsigned total_discard = 2*overlap;
  unsigned step = ndat - total_discard;

  unsigned in_total_ndat = npart * step + total_discard;
  unsigned out_total_ndat = npart * ndat;

  unsigned in_size = npol * nchan * in_total_ndat;
  unsigned out_size = npol * nchan * out_total_ndat;
  unsigned apod_size = ndat;

  // generate some data.
  std::vector< std::complex<float> > in(in_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > apod_cpu(apod_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > apod_gpu(apod_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  unsigned in_idx;
  float val;

  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      val = ichan*npol + ipol;
      for (unsigned ipart=0; ipart<npart; ipart++) {
        for (unsigned idat=0; idat<step; idat++) {
          in_idx = ichan*npol*in_total_ndat + ipol*in_total_ndat + ipart*step + idat;
          in[in_idx] = std::complex<float>(val, val);
          val++;
        }
      }
    }
  }


  std::vector<unsigned> in_dim = {nchan, npol, in_total_ndat};
  std::vector<unsigned> out_dim = {nchan, npol, out_total_ndat};

  // apodization filter is just multiplying by 2.
  auto random_gen = test::util::random<float>();
  for (unsigned i=0; i<apod_size; i++) {
    apod_cpu[i] = std::complex<float>(random_gen(), 0.0);
    apod_gpu[i] = std::complex<float>(apod_cpu[i].real(), apod_cpu[i].real());
  }

  auto t = test::util::now();
  test::util::InverseFilterbank::apodization_overlap_cpu_FPT< std::complex<float> >(
    in, apod_cpu, out_cpu, overlap, npart, npol, nchan, ndat, in_total_ndat, out_total_ndat
  );


  test::util::delta<std::milli> delta_cpu = test::util::now() - t;

  t = test::util::now();
  CUDA::InverseFilterbankEngineCUDA::apply_k_apodization_overlap(
    in, apod_gpu, out_gpu, overlap, npart, npol, nchan, ndat
  );
  test::util::delta<std::milli> delta_gpu = test::util::now() - t;

  if (test::util::config::verbose) {
    std::cerr << "apodization overlap GPU: " << delta_gpu.count()
      << " ms; CPU: " << delta_cpu.count()
      << " ms; CPU/GPU: " << delta_cpu.count() / delta_gpu.count()
      << std::endl;
  }

  bool allclose = test::util::allclose(out_cpu, out_gpu, thresh[0]);
  REQUIRE(allclose == true);
}

TEST_CASE (
  "reponse stitch kernel should produce expected output",
  "[no_file][cuda][response_stitch]"
)
{
  std::vector<float> thresh = test_config.get_thresh();
  std::vector<test::util::TestShape> test_shapes = test_config.get_test_vector_shapes();
  auto idx = GENERATE_COPY(range(0, (int) test_shapes.size()));

  test::util::TestShape test_shape = test_shapes[idx];

  auto rand_gen = test::util::random<float>();

  unsigned npart = test_shape.npart;
  unsigned npol = test_shape.input_npol;
  unsigned nchan = test_shape.input_nchan;
  unsigned ndat = test_shape.input_ndat;

  Rational os_factor(4, 3);
  unsigned in_ndat_keep = os_factor.normalize(ndat);
  unsigned out_ndat = nchan * in_ndat_keep;
  unsigned out_size = out_ndat * npol * npart;
  unsigned in_size = ndat * nchan * npol * npart;
  // generate some data.
  std::vector<std::complex<float>> in(in_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> resp(out_ndat, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  unsigned in_idx;
  float val;
  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      val = ichan;
      for (unsigned ipart=0; ipart<npart; ipart++) {
        for (unsigned idat=0; idat<ndat; idat++) {
          in_idx = ichan*npol*npart*ndat + ipol*npart*ndat + ipart*ndat + idat;
          // in_idx = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat + idat;
          in[in_idx] = std::complex<float>(val, val);
          val += 1;
        }
      }
    }
  }

  std::vector<unsigned> in_dim = {nchan, npol, npart*ndat};
  std::vector<unsigned> out_dim = {1, npol, npart*out_ndat};

  // test::util::print_array(in, in_dim);

  // response is just multiplying by 2.
  for (unsigned i=0; i<out_ndat; i++) {
    resp[i] = std::complex<float>(rand_gen(), 0.0);
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

      auto t = test::util::now();
      test::util::InverseFilterbank::response_stitch_cpu_FPT<std::complex<float>>(
        in, resp, out_cpu, os_factor, npart, npol, nchan, ndat, *dc_it, *all_it
      );
      test::util::delta<std::milli> delta_cpu = test::util::now() - t;
      // test::util::print_array(out_cpu, out_dim);
      t = test::util::now();
      CUDA::InverseFilterbankEngineCUDA::apply_k_response_stitch(
        in, resp, out_gpu, os_factor, npart, npol, nchan, ndat, *dc_it, *all_it
      );

      test::util::delta<std::milli> delta_gpu = test::util::now() - t;

      if (*dc_it && *all_it && test::util::config::verbose) {
        std::cerr << "response stitch gpu: " << delta_gpu.count()
          << " ms; CPU: " << delta_cpu.count()
          << " ms; CPU/GPU: " << delta_cpu.count() / delta_gpu.count()
          << std::endl;
      }
      // test::util::print_array(out_gpu, out_dim);
      // test::util::print_array(out_cpu, out_dim);

      allclose = test::util::allclose<std::complex<float>>(out_cpu, out_gpu, thresh[0]);

      REQUIRE(allclose == true);
    }
  }
}
