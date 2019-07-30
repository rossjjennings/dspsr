#include <cstdlib>
#include <string>

#include "dsp/Shape.h"

#include "util.hpp"

void check_error (const char*);


std::chrono::time_point<std::chrono::high_resolution_clock> util::now ()
{
  return std::chrono::high_resolution_clock::now();
}

void util::load_psr_data (dsp::IOManager manager, unsigned block_size, dsp::TimeSeries* ts)
{
  dsp::Input* input = manager.get_input();
  input->set_block_size(block_size);
  while (! manager.get_input()->eod()) {
    manager.load(ts);
  }
}

void util::set_verbose (bool val)
{
  dsp::Input::verbose = val;
  dsp::Operation::verbose = val;
  dsp::Observation::verbose = val;
  dsp::Shape::verbose = val;
  util::verbose = val;
}


std::string util::get_test_data_dir ()
{
  const char* env_data_dir = std::getenv("DSPSR_TEST_DATA_DIR");
  if (env_data_dir) {
    return std::string(env_data_dir);
  } else {
    return ".";
  }
}



std::function<void(dsp::TimeSeries*, dsp::TimeSeries*, cudaMemcpyKind)> util::transferTimeSeries (
  cudaStream_t stream, CUDA::DeviceMemory* memory
)
{
  return [stream, memory] (dsp::TimeSeries* in, dsp::TimeSeries* out, cudaMemcpyKind k){
    dsp::TransferCUDA transfer(stream);
    transfer.set_kind(k);
    if (k == cudaMemcpyHostToDevice) {
      out->set_memory(memory);
    }
    transfer.set_input(in);
    transfer.set_output(out);
    transfer.prepare();
    transfer.operate();
  };
}

bool util::allclose (dsp::TimeSeries* a, dsp::TimeSeries* b, float atol, float rtol)
{
  bool allclose = true;

  std::vector<std::string> shape_str = {"nchan", "npol", "ndat", "ndim"};
  std::vector<unsigned> a_shape = {
      a->get_nchan(), a->get_npol(), a->get_ndat(), a->get_ndim()
  };

  std::vector<unsigned> b_shape = {
      b->get_nchan(), b->get_npol(), b->get_ndat(), b->get_ndim()
  };

  for (unsigned idx=0; idx<shape_str.size(); idx++) {
    if (a_shape[idx] != b_shape[idx])
    {
      std::cerr << "util::allclose: " << shape_str[idx] << " dim not equal: "<< a_shape[idx] << " != " << b_shape[idx] << std::endl;
      return false;
    }
  }

  float* a_ptr;
  float* b_ptr;

  for (unsigned ichan=0; ichan<a->get_nchan(); ichan++) {
    for (unsigned ipol=0; ipol<a->get_npol(); ipol++) {
      std::cerr << "(ichan, ipol) = (" << ichan << ", " << ipol << ")" << std::endl;
      a_ptr = a->get_datptr(ichan, ipol);
      b_ptr = b->get_datptr(ichan, ipol);
      for (unsigned idat=0; idat<a->get_ndat()*a->get_ndim(); idat++) {
        if (idat < 100) {
          std::cerr << "(" << *a_ptr << ", " << *b_ptr << ") ";          
        }
        if (! util::isclose(*a_ptr, *b_ptr, atol, rtol)) {
          // std::cerr << "util::allclose: ipol=" << ipol << ", ichan=" << ichan << std::endl;
          allclose = false;
          // break;
        }
        a_ptr++;
        b_ptr++;
      }
      std::cerr << std::endl;
    }
  }
  return allclose;
}
