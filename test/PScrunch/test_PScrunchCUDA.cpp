
#include "config.h"
#include "catch.hpp"

#include "dsp/PScrunch.h"

#include "dsp/PScrunchCUDA.h"
#include "dsp/MemoryCUDA.h"

#include "util/util.hpp"

#include <iostream>
#include <functional>

using std::cerr;
using std::endl;

TEST_CASE(
  "PScrunchCUDA",
  "[cuda][unit][no_file][PScrunch]"
)
{
  SECTION ("Can construct PScrunchCUDA object") {
    void* stream = 0;
    int device = 0;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    Reference::To<dsp::Memory> device_memory = new CUDA::DeviceMemory (cuda_stream, device);
    CUDA::PScrunchEngine engine(cuda_stream);
  }
}

TEST_CASE(
    "PScrunchCUDA::setup",
  "[cuda][component][no_file][PScrunch]"
)
{
  void* stream = 0;
  int device = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory * device_memory = new CUDA::DeviceMemory (cuda_stream, device);

  CUDA::PScrunchEngine engine(cuda_stream);
  dsp::PScrunch pscrunch;

  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> dinput = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> doutput = new dsp::TimeSeries;

  auto transfer = test::util::transferTimeSeries(cuda_stream, device_memory);

  // nchan, npol, ndat
  std::vector<unsigned> shape {
      8, 2, 100
  };

  input->set_state(Signal::Coherence);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_ndim(1);
  input->resize(shape[2]);

  transfer(input, dinput, cudaMemcpyHostToDevice);
  doutput->set_memory(device_memory);

  pscrunch.set_engine(&engine);
  pscrunch.set_input(input);
  pscrunch.set_output(output);
  pscrunch.set_output_state(Signal::Intensity);

  SECTION("Can call prepare")
  {
    pscrunch.prepare();
  }
}

TEST_CASE(
  "PScrunchCUDA::operate input npol=2",
  "[cuda][component][no_file][PScrunch]"
)
{
  void* stream = 0;
  int device = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory * device_memory = new CUDA::DeviceMemory (cuda_stream, device);

  dsp::PScrunch pscrunch;

  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> dinput = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> doutput = new dsp::TimeSeries;
  
  // nchan, npol, ndat
  std::vector<unsigned> shape {
      8, 2, 100
  };

  auto filler_fpt = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (ichan*shape[1]*shape[2] + ipol*shape[2] + idat);
  };

  auto filler_tfp = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (idat*shape[0]*shape[1] + ichan*shape[1] + ipol);
  };

  auto intensity_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (filler_fpt(ichan, 0, idat, 0) + filler_fpt(ichan, 1, idat, 0));
  };

  auto intensity_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (filler_tfp(ichan, 0, idat, 0) + filler_tfp(ichan, 1, idat, 0));
  };

  auto pp_state_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, 0, idat, 0);
  };

  auto pp_state_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, 0, idat, 0);
  };

  auto qq_state_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, 1, idat, 0);
  };

  auto qq_state_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, 1, idat, 0);
  };

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map_fpt;
  map_fpt[Signal::Intensity] = intensity_fpt;
  map_fpt[Signal::PP_State] = pp_state_fpt;
  map_fpt[Signal::QQ_State] = qq_state_fpt;

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map_tfp;
  map_tfp[Signal::Intensity] = intensity_tfp;
  map_tfp[Signal::PP_State] = pp_state_tfp;
  map_tfp[Signal::QQ_State] = qq_state_tfp;

  input->set_ndim(1);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_state(Signal::PPQQ);
  input->set_order(dsp::TimeSeries::Order::OrderFPT);
  input->resize(shape[2]);

  auto transfer = test::util::transferTimeSeries(cuda_stream, device_memory);
  transfer(input, dinput, cudaMemcpyHostToDevice);
  doutput->set_memory(device_memory);

  pscrunch.set_output_state(Signal::Intensity);
  pscrunch.set_input(dinput);
  pscrunch.set_output(doutput);
  pscrunch.set_engine(new CUDA::PScrunchEngine(cuda_stream));

  std::vector<Signal::State> output_states = { Signal::PP_State, Signal::QQ_State, Signal::Intensity };

  test::util::init_TimeSeries(input, filler_fpt);
  transfer(input, dinput, cudaMemcpyHostToDevice);

  for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map_fpt.begin(); ot != map_fpt.end(); ++ot)
  {
    pscrunch.set_output_state(ot->first);
    pscrunch.prepare();
    pscrunch.operate();
    transfer(doutput, output, cudaMemcpyDeviceToHost);
    uint64_t nerrors = test::util::check_TimeSeries(output, ot->second);
    CHECK(nerrors == 0);
  }

  input->set_order(dsp::TimeSeries::Order::OrderTFP);
  input->resize(shape[2]);

  test::util::init_TimeSeries_TFP(input, filler_tfp);
  transfer(input, dinput, cudaMemcpyHostToDevice);

  for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map_tfp.begin(); ot != map_tfp.end(); ++ot)
  {
    pscrunch.set_output_state(ot->first);
    pscrunch.prepare();
    pscrunch.operate();
    transfer(doutput, output, cudaMemcpyDeviceToHost);
    uint64_t nerrors = test::util::check_TimeSeries_TFP(output, ot->second);
    CHECK(nerrors == 0);
  }
}


TEST_CASE(
  "PScrunchCUDA::operate input npol=4",
  "[cuda][component][no_file][PScrunch]"
)
{
  void* stream = 0;
  int device = 0;
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  CUDA::DeviceMemory * device_memory = new CUDA::DeviceMemory (cuda_stream, device);

  dsp::PScrunch pscrunch;
  Reference::To<dsp::TimeSeries> input = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> output = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> dinput = new dsp::TimeSeries;
  Reference::To<dsp::TimeSeries> doutput = new dsp::TimeSeries;

  // nchan, npol, ndat
  std::vector<unsigned> shape {
      8, 4, 100
  };

  auto filler_fpt = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (ichan*shape[1]*shape[2] + ipol*shape[2] + idat);
  };

  auto filler_tfp = [&shape] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (float) (idat*shape[0]*shape[1] + ichan*shape[1] + ipol);
  };

  auto ppqq_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, ipol, idat, 0);
  };

  auto ppqq_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, ipol, idat, 0);
  };

  auto intensity_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (filler_fpt(ichan, 0, idat, 0) + filler_fpt(ichan, 1, idat, 0));
  };

  auto intensity_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return (filler_tfp(ichan, 0, idat, 0) + filler_tfp(ichan, 1, idat, 0));
  };

  auto pp_state_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, 0, idat, 0);
  };

  auto pp_state_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, 0, idat, 0);
  };

  auto qq_state_fpt = [&shape, &filler_fpt] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_fpt(ichan, 1, idat, 0);
  };

  auto qq_state_tfp = [&shape, &filler_tfp] (unsigned ichan, unsigned ipol, unsigned idat, unsigned idim) -> float
  {
    return filler_tfp(ichan, 1, idat, 0);
  };

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map_fpt;
  map_fpt[Signal::PPQQ] = ppqq_fpt;
  map_fpt[Signal::Intensity] = intensity_fpt;
  map_fpt[Signal::PP_State] = pp_state_fpt;
  map_fpt[Signal::QQ_State] = qq_state_fpt;

  std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> > map_tfp;
  map_tfp[Signal::PPQQ] = ppqq_tfp;
  map_tfp[Signal::Intensity] = intensity_tfp;
  map_tfp[Signal::PP_State] = pp_state_tfp;
  map_tfp[Signal::QQ_State] = qq_state_tfp;

  input->set_ndim(1);
  input->set_nchan(shape[0]);
  input->set_npol(shape[1]);
  input->set_ndat(shape[2]);
  input->set_state(Signal::Coherence);
  input->set_order(dsp::TimeSeries::Order::OrderFPT);
  input->resize(shape[2]);

  auto transfer = test::util::transferTimeSeries(cuda_stream, device_memory);
  transfer(input, dinput, cudaMemcpyHostToDevice);
  doutput->set_memory(device_memory);

  pscrunch.set_output_state(Signal::Intensity);
  pscrunch.set_input(dinput);
  pscrunch.set_output(doutput);
  pscrunch.set_engine(new CUDA::PScrunchEngine(cuda_stream));

  std::vector<Signal::State> output_states = { Signal::PP_State, Signal::QQ_State, Signal::Intensity };

  test::util::init_TimeSeries(input, filler_fpt);
  transfer(input, dinput, cudaMemcpyHostToDevice);

  for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map_fpt.begin(); ot != map_fpt.end(); ++ot)
  {
    pscrunch.set_output_state(ot->first);
    pscrunch.prepare();
    pscrunch.operate();
    transfer(doutput, output, cudaMemcpyDeviceToHost);
    uint64_t nerrors = test::util::check_TimeSeries(output, ot->second);
    CHECK(nerrors == 0);
  }

  input->set_order(dsp::TimeSeries::Order::OrderTFP);
  input->resize(shape[2]);

  test::util::init_TimeSeries_TFP(input, filler_tfp);
  transfer(input, dinput, cudaMemcpyHostToDevice);

  for (std::map<Signal::State, std::function<float(unsigned, unsigned, unsigned, unsigned)> >::iterator ot = map_tfp.begin(); ot != map_tfp.end(); ++ot)
  {
    pscrunch.set_output_state(ot->first);
    pscrunch.prepare();
    pscrunch.operate();
    transfer(doutput, output, cudaMemcpyDeviceToHost);
    uint64_t nerrors = test::util::check_TimeSeries_TFP(output, ot->second);
    CHECK(nerrors == 0);
  }
}
