#include <complex>
#include <vector>
#include <iostream>
#include <algorithm> // for all_of, for_each

#include "catch.hpp"

#include "util.hpp"
#include "util.cuda.hpp"

#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/InverseFilterbank.h"
#include "Rational.h"

const float thresh = 1e-5;

enum TestTypeEnum {SmallSinglePol, SmallDoublePol, SinglePol, DoublePol};

template<unsigned T>
struct TestDims;

template<>
struct TestDims<SmallSinglePol> {
  static const unsigned npart = 3;
  static const unsigned npol = 1;
  static const unsigned nchan = 4;
  static const unsigned output_nchan = 2;
  static const unsigned ndat = 8;
  static const unsigned overlap = 1;
};


template<>
struct TestDims<SmallDoublePol> : TestDims<SmallSinglePol> {
  static const unsigned npol = 2;
};

template<>
struct TestDims<SinglePol> {
  static const unsigned npart = 10;
  static const unsigned npol = 1;
  static const unsigned nchan = 64;
  static const unsigned output_nchan = 8;
  static const unsigned ndat = 256;
  static const unsigned overlap = 32;
};

template<>
struct TestDims<DoublePol> : TestDims<SinglePol> {
  static const unsigned npol = 2;
};


TEST_CASE ("InverseFilterbankEngineCUDA unit", "") {

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA::InverseFilterbankEngineCUDA engine(cuda_stream);

  auto pred_setup_fft_plan = [] (cufftResult r) -> bool { return r == CUFFT_SUCCESS; };

  SECTION ("can create forward cufft plan")
  {
    std::vector<cufftResult> results = engine.setup_forward_fft_plan(
      1024, 8, CUFFT_R2C);
    REQUIRE(all_of(results.begin(), results.end(), pred_setup_fft_plan) == true);

    results = engine.setup_forward_fft_plan(1024, 8, CUFFT_C2C);
    REQUIRE(all_of(results.begin(), results.end(), pred_setup_fft_plan) == true);

  }

  SECTION ("can create backward cufft plan")
  {
    std::vector<cufftResult> results = engine.setup_backward_fft_plan(1024, 8);
    REQUIRE(all_of(results.begin(), results.end(), pred_setup_fft_plan) == true);
  }
}

TEMPLATE_TEST_CASE (
  "InverseFilterbankEngineCUDA component",
  "[InverseFilterbankEngineCUDA][template]",
  TestDims<SinglePol>
)
{

  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA::InverseFilterbankEngineCUDA engine(cuda_stream);
  Rational os_factor (4, 3);

  unsigned npart = TestType::npart;
  unsigned npol = TestType::npol;
  unsigned input_nchan = TestType::nchan;
  unsigned output_nchan = TestType::output_nchan;

  auto os_in2out = [input_nchan, output_nchan, os_factor] (unsigned n) -> unsigned {
    return os_factor.normalize(n) * input_nchan / output_nchan;
  };

  unsigned input_fft_length = TestType::ndat;
  unsigned output_fft_length = os_in2out(input_fft_length);
  unsigned input_overlap = TestType::overlap;
  unsigned output_overlap = os_in2out(input_overlap);

  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;

  std::vector<unsigned> in_dim = {
    input_nchan, npol, input_fft_length*npart};
  std::vector<unsigned> out_dim = {
    output_nchan, npol, output_fft_length*npart};
  unsigned in_size = 1;
  std::for_each (in_dim.begin(), in_dim.end(), [&in_size](unsigned i){in_size*=i;});
  unsigned out_size = 1;
  std::for_each (out_dim.begin(), out_dim.end(), [&out_size](unsigned i){out_size*=i;});
  std::vector<std::complex<float>> in_vec(in_size);
  std::vector<std::complex<float>> out_vec(out_size);
  for (unsigned idx=0; idx<in_size; idx++) {
    in_vec[idx] = std::complex<float>(idx, idx);
  }
  util::loadTimeSeries<std::complex<float>>(in_vec, input, in_dim);
  util::loadTimeSeries<std::complex<float>>(out_vec, output, out_dim);

  SECTION ("can call setup method")
  {
    engine.setup(
      input, output, os_factor, input_fft_length, output_fft_length,
      input_overlap, input_overlap, output_overlap, output_overlap, true, true
    );
  }

  SECTION ("can call perform method")
  {
    engine.setup(
      input, output, os_factor, input_fft_length, output_fft_length,
      input_overlap, input_overlap, output_overlap, output_overlap, true, true
    );
    engine.perform(
      input, output, npart
    );
  }
}



TEST_CASE ("cufft kernels can operate on data", "")
{
  void* stream = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  CUDA::InverseFilterbankEngineCUDA engine(cuda_stream);

  unsigned fft_length = 1024;
  unsigned nchan = 8;
  unsigned ndat = fft_length * nchan;

  SECTION (
    "forward FFT kernel can operate on data"
  )
  {

    std::vector<std::complex<float>> in_c (ndat);
    std::vector<float> in_r (ndat);
    std::vector<std::complex<float>> out (ndat);

    engine.setup_forward_fft_plan(fft_length, nchan, CUFFT_R2C);
    engine.apply_cufft_forward<CUFFT_R2C>(in_r, out);
    engine.setup_forward_fft_plan(fft_length, nchan, CUFFT_C2C);
    engine.apply_cufft_forward<CUFFT_C2C>(in_c, out);
  }

  SECTION (
    "backward FFT kernel can operate on data"
  )
  {
    std::vector<std::complex<float>> in (ndat);
    std::vector<std::complex<float>> out (ndat);
    engine.setup_backward_fft_plan(fft_length, nchan);
    engine.apply_cufft_backward(in, out);
  }
}

TEMPLATE_TEST_CASE (
  "output overlap discard kernel should produce expected output",
  "[overlap_discard][template]",
  TestDims<SmallSinglePol>, TestDims<SmallDoublePol>, TestDims<SinglePol>, TestDims<DoublePol>
)
{
  unsigned npart = TestType::npart;
  unsigned npol = TestType::npol;
  unsigned nchan = TestType::nchan;
  unsigned ndat = TestType::ndat;
  unsigned overlap = TestType::overlap;

  unsigned out_ndat = ndat - 2*overlap;

  unsigned in_size = npart * npol * nchan * ndat;
  unsigned out_size = npart * npol * nchan * out_ndat;

  // generate some data.
  std::vector<std::complex<float>> in(in_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector<std::complex<float>> out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  unsigned idx;
  float val;

  for (unsigned ipart=0; ipart<npart; ipart++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      val = ipol + npol*ipart;
      for (unsigned ichan=0; ichan<nchan; ichan++) {
        for (unsigned idat=0; idat<ndat; idat++) {
          idx = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat + idat;
          in[idx] = std::complex<float>(val, val);
          val++;
        }
      }
    }
  }
  std::vector<unsigned> in_dim = {npart, npol, nchan, ndat};
  std::vector<unsigned> out_dim = {npart, npol, nchan, out_ndat};

  auto t = util::now();
  util::overlap_discard_cpu(
    in, out_cpu, overlap, npart, npol, nchan, ndat
  );
  util::delta<std::milli> delta_cpu = util::now() - t;

  t = util::now();
  CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard(
    in, out_gpu, overlap, npart, npol, nchan, ndat
  );

  // util::prunsigned_array<std::complex<float>>(in, in_dim);
  // util::prunsigned_array<std::complex<float>>(out_cpu, out_dim);
  // util::prunsigned_array<std::complex<float>>(out_gpu, out_dim);

  util::delta<std::milli> delta_gpu = util::now() - t;
  std::cerr << "overlap discard GPU: " << delta_gpu.count()
    << " ms; CPU: " << delta_cpu.count()
    << " ms; CPU/GPU: " << delta_cpu.count() / delta_gpu.count()
    << std::endl;

  bool allclose = util::allclose(out_cpu, out_gpu, thresh);
  REQUIRE(allclose == true);
}



TEMPLATE_TEST_CASE (
  "apodization overlap kernel should produce expected output",
  "[apodization_overlap][template]",
  TestDims<SmallSinglePol>, TestDims<SmallDoublePol>, TestDims<SinglePol>, TestDims<DoublePol>
)
{
  unsigned npart = TestType::npart;
  unsigned npol = TestType::npol;
  unsigned nchan = TestType::nchan;
  unsigned ndat = TestType::ndat;
  unsigned overlap = TestType::overlap;

  unsigned out_ndat = ndat - 2*overlap;

  unsigned in_size = npart * npol * nchan * ndat;
  unsigned out_size = npart * npol * nchan * out_ndat;
  unsigned apod_size = nchan * out_ndat;

  // generate some data.
  std::vector< std::complex<float> > in(in_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > apod(apod_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > out_cpu(out_size, std::complex<float>(0.0, 0.0));
  std::vector< std::complex<float> > out_gpu(out_size, std::complex<float>(0.0, 0.0));

  // fill up input data such that each time sample in each channel has the same value.
  unsigned idx;
  float val;

  for (unsigned ipart=0; ipart<npart; ipart++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      val = ipol + npol*ipart;
      for (unsigned ichan=0; ichan<nchan; ichan++) {
        for (unsigned idat=0; idat<ndat; idat++) {
          idx = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat + idat;
          in[idx] = std::complex<float>(val, val);
          val++;
        }
      }
    }
  }
  std::vector<unsigned> in_dim = {npart, npol, nchan, ndat};
  std::vector<unsigned> out_dim = {npart, npol, nchan, out_ndat};

  // apodization filter is just multiplying by 2.
  for (unsigned i=0; i<apod_size; i++) {
    apod[i] = std::complex<float>(2.0, 0.0);
  }

  auto t = util::now();
  util::apodization_overlap_cpu< std::complex<float> >(
    in, apod, out_cpu, overlap, npart, npol, nchan, ndat
  );
  util::delta<std::milli> delta_cpu = util::now() - t;

  t = util::now();
  CUDA::InverseFilterbankEngineCUDA::apply_k_apodization_overlap(
    in, apod, out_gpu, overlap, npart, npol, nchan, ndat
  );
  util::delta<std::milli> delta_gpu = util::now() - t;

  std::cerr << "apodization overlap GPU: " << delta_gpu.count()
    << " ms; CPU: " << delta_cpu.count()
    << " ms; CPU/GPU: " << delta_cpu.count() / delta_gpu.count()
    << std::endl;

  bool allclose = util::allclose(out_cpu, out_gpu, thresh);
  REQUIRE(allclose == true);
}

TEMPLATE_TEST_CASE (
  "reponse stitch kernel should produce expected output",
  "[response_stitch][template]",
  TestDims<SmallSinglePol>, TestDims<SmallDoublePol>, TestDims<SinglePol>, TestDims<DoublePol>
)
{
  unsigned npart = TestType::npart;
  unsigned npol = TestType::npol;
  unsigned nchan = TestType::nchan;
  unsigned ndat = TestType::ndat;

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
  for (unsigned ipart=0; ipart<npart; ipart++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      val = 0;
      for (unsigned ichan=0; ichan<nchan; ichan++) {
        for (unsigned idat=0; idat<ndat; idat++) {
          val += 1;
          in_idx = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat + idat;
          in[in_idx] = std::complex<float>(val, val);
        }
      }
    }
  }

  std::vector<unsigned> dim_in = {npart, npol, nchan, ndat};
  std::vector<unsigned> dim_out = {npart, npol, out_ndat};

  // util::prunsigned_array<std::complex<float>>(in, dim_in);

  // response is just multiplying by 2.
  for (unsigned i=0; i<out_ndat; i++) {
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

      auto t = util::now();
      util::response_stitch_cpu<std::complex<float>>(
        in, resp, out_cpu, os_factor, npart, npol, nchan, ndat, *dc_it, *all_it
      );
      util::delta<std::milli> delta_cpu = util::now() - t;

      t = util::now();
      CUDA::InverseFilterbankEngineCUDA::apply_k_response_stitch(
        in, resp, out_gpu, os_factor, npart, npol, nchan, ndat, *dc_it, *all_it
      );
      util::delta<std::milli> delta_gpu = util::now() - t;

      if (*dc_it && *all_it) {
        std::cerr << "response stitch GPU: " << delta_gpu.count()
          << " ms; CPU: " << delta_cpu.count()
          << " ms; CPU/GPU: " << delta_cpu.count() / delta_gpu.count()
          << std::endl;
      }

      allclose = util::allclose<std::complex<float>>(out_cpu, out_gpu, thresh);

      REQUIRE(allclose == true);
    }
  }
}
