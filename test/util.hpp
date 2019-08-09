#ifndef __util_hpp
#define __util_hpp

#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <chrono>
#include <algorithm>
#include <time.h>
#include <cstdlib>

// #include "json.hpp"
#include "toml.h"

#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDA.h"
#include "dsp/Scratch.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "Rational.h"

// using json = nlohmann::json;

void check_error (const char*);

namespace util {

  struct TestShape {
    unsigned npart;

    unsigned input_nchan;
    unsigned output_nchan;

    unsigned input_npol;
    unsigned output_npol;

    unsigned input_ndat;
    unsigned output_ndat;

    unsigned overlap_pos;
    unsigned overlap_neg;
  };

  void from_toml (const toml::Value& val, TestShape& sh);

  // void to_json(json& j, const TestShape& sh);
  //
  // void from_json(const json& j, TestShape& sh);

  struct config {
    static bool verbose;
  };

  template<typename T>
  struct std2dspsr;

  template<>
  struct std2dspsr<float> {
    static const Signal::State state = Signal::Nyquist;
    static const unsigned ndim = 1;
  };

  template<>
  struct std2dspsr<std::complex<float>> {
    static const Signal::State state = Signal::Analytic;
    static const unsigned ndim = 2;
  };

  template<typename unit>
  using delta = std::chrono::duration<double, unit>;

  std::chrono::time_point<std::chrono::high_resolution_clock> now ();

  template<typename T>
  bool isclose (T a, T b, float atol=1e-7, float rtol=1e-5);

  template<typename T>
  bool allclose (const T* a, const T* b, unsigned size, float atol=1e-7, float rtol=1e-5);

  template<typename T>
  bool allclose (const std::vector<T>& a, const std::vector<T>& b, float atol=1e-7, float rtol=1e-5);

  bool allclose (dsp::TimeSeries* a, dsp::TimeSeries* b, float atol=1e-7, float rtol=1e-5);

  template<typename T>
  unsigned nclose (const T* a, const T* b, unsigned size, float atol=1e-7, float rtol=1e-5);

  template<typename T>
  unsigned nclose (const std::vector<T>& a, const std::vector<T>& b, float atol=1e-7, float rtol=1e-5);


  template<typename T>
  void load_binary_data (std::string file_path, std::vector<T>& test_data);

  template<typename T>
  void write_binary_data (std::string file_path, std::vector<T>& data);

  template<typename T>
  void write_binary_data (std::string file_path, T* buffer, unsigned len);

  // json load_json (std::string file_path);
  const toml::Value load_toml (const std::string& file_path);

  template<typename T>
  void print_array (const std::vector<T>& arr, std::vector<unsigned>& dim);

  template<typename T>
  void print_array (T* arr, std::vector<unsigned>& dim);

  template<typename T>
  std::function<T(void)> random ();

  template<typename T>
  T sum (std::vector<T>);

  template<typename T>
  T product (std::vector<T>);

  void load_psr_data (dsp::IOManager manager, unsigned block_size, dsp::TimeSeries* ts);

  void set_verbose (bool val);

  std::string get_test_env_var (const std::string& env_var_name,const std::string& default_val = ".");

  std::string get_test_data_dir ();

  //! load data from a stl vector unsignedo a dspsr TimeSeries object.
  //! dim should be of length 3.
  //! Assumes TimeSeries is in FPT order.
  template<typename T>
  void loadTimeSeries (
    const std::vector<T>& in,
    dsp::TimeSeries* out,
    const std::vector<unsigned>& dim);

  std::function<void(dsp::TimeSeries*, dsp::TimeSeries*, cudaMemcpyKind)> transferTimeSeries (
    cudaStream_t stream, CUDA::DeviceMemory* memory);

  template<class FilterbankType>
  class IntegrationTestConfiguration {

  public:

    IntegrationTestConfiguration (
      const Rational& _os_factor,
      unsigned _npart,
      unsigned _npol,
      unsigned _input_nchan,
      unsigned _output_nchan,
      unsigned _input_ndat,
      unsigned _input_overlap
    ) : os_factor(_os_factor),
        npart(_npart),
        npol(_npol),
        input_nchan(_input_nchan),
        output_nchan(_output_nchan),
        input_ndat(_input_ndat),
        input_overlap(_input_overlap)
    {
      scratch = new dsp::Scratch;
      filterbank = new FilterbankType;
    }

    void setup (
      dsp::TimeSeries* in,
      dsp::TimeSeries* out
    );

    template<class MemoryType>
    std::vector<float*> allocate_scratch(
      MemoryType* _memory=nullptr
    );

    Reference::To<dsp::Scratch> scratch;
    Reference::To<FilterbankType> filterbank;

  private:

    unsigned total_scratch_needed;
    unsigned stitching_scratch_space;
    unsigned overlap_discard_scratch_space;

    const Rational& os_factor;
    unsigned npart;
    unsigned npol;
    unsigned input_nchan;
    unsigned output_nchan;
    unsigned input_ndat;
    unsigned input_overlap;

  };
}

template<typename T>
void util::load_binary_data (std::string file_path, std::vector<T>& test_data)
{
  std::streampos size;

  std::ifstream file (file_path, std::ios::in|std::ios::binary|std::ios::ate);
  if (file.is_open())
  {
    file.seekg(0, std::ios::end);
    size = file.tellg();
    file.seekg(0, std::ios::beg);

    // read the data:
    std::vector<char> file_bytes(size);
    file.read(&file_bytes[0], size);
    file.close();

    unsigned T_size = (size / sizeof(T));
    // std::cerr << "T_size=" << T_size << std::endl;

    const T* data = reinterpret_cast<const T*>(file_bytes.data());
    test_data.assign(data, data + T_size);
  }
}

template<typename T>
void util::write_binary_data (std::string file_path, std::vector<T> buffer)
{
  util::write_binary_data(file_path, buffer.data(), buffer.size());
}

template<typename T>
void util::write_binary_data (std::string file_path, T* buffer, unsigned len)
{
  std::ofstream file(file_path, std::ios::out | std::ios::binary);

  file.write(
    reinterpret_cast<const char*>(buffer),
    len*sizeof(T)
  );
  file.close();
}


template<typename T>
unsigned util::nclose (const std::vector<T>& a, const std::vector<T>& b, float atol, float rtol) {

  if (a.size() != b.size()) {
    std::string msg ("util::nclose: vectors not the same size");
    if (util::config::verbose) {
     std::cerr << msg << std::endl;
    }
    throw msg;
  }
  if (util::config::verbose) {
    std::cerr << "util::nclose size=" << a.size() << std::endl;
  }
  return util::nclose(a.data(), b.data(), (unsigned) a.size(), atol, rtol);
}


template<typename T>
unsigned util::nclose (const T* a, const T* b, unsigned size, float atol, float rtol)
{
  unsigned nclose=0;
  bool close;
  for (unsigned i=0; i<size; i++) {
    close = util::isclose<T>(a[i], b[i], atol, rtol);
    if (close) {
      nclose++;
    }
  }
  return nclose;
}


template<typename T>
bool util::allclose (const std::vector<T>& a, const std::vector<T>& b, float atol, float rtol) {

  return util::nclose<T> (a, b, atol, rtol) == a.size();
}


template<typename T>
bool util::allclose (const T* a, const T* b, unsigned size, float atol, float rtol)
{
  return util::nclose<T> (a, b, size, atol, rtol) == size;
}

template<typename T>
bool util::isclose (T a, T b, float atol, float rtol)
{
  return abs(a - b) <= (atol + rtol*abs(b));
}

template<typename T>
void util::print_array (const std::vector<T>& arr, std::vector<unsigned>& dim)
{
  util::print_array<T>(const_cast<T*>(arr.data()), dim);
}

template<typename T>
void util::print_array (T* arr, std::vector<unsigned>& dim)
{
  if (dim.size() > 2) {
    unsigned head_dim = dim[0];
    std::vector<unsigned> tail_dim(dim.begin() + 1, dim.end());
    unsigned stride = 1;
    for (unsigned d=1; d<dim.size(); d++) {
      stride *= dim[d];
    }
    for (unsigned i=0; i<head_dim; i++) {
      util::print_array<T>(arr + stride*i, tail_dim);
    }
  } else {
    for (unsigned i=0; i<dim[0]; i++) {
      for (unsigned j=0; j<dim[1]; j++) {
        std::cerr << arr[i*dim[1] + j] << " ";
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl;
  }
}

template<typename T>
T util::sum (std::vector<T> a)
{
  T res = 0;
  std::for_each (a.begin(), a.end(), [&res](unsigned i){res+=i;});
  return res;
}

template<typename T>
T util::product (std::vector<T> a)
{
  T res = 1;
  std::for_each (a.begin(), a.end(), [&res](unsigned i){res*=i;});
  return res;
}

template<typename T>
std::function<T(void)> util::random ()
{
  srand (time(NULL));
  return [] () { return (T) rand() / RAND_MAX; };
}


template<typename T>
void util::loadTimeSeries (
  const std::vector<T>& in,
  dsp::TimeSeries* out,
  const std::vector<unsigned>& dim
)
{
  typedef util::std2dspsr<T> dspsr_type;

  unsigned ndim = dspsr_type::ndim;

  out->set_state (dspsr_type::state);

  out->set_nchan (dim[0]);
  out->set_npol (dim[1]);
  out->set_ndat (dim[2]);
  out->set_ndim (ndim);
  out->resize (dim[2]);

  if (util::config::verbose) {
    std::cerr << "util::loadTimeSeries: ("
      << dim[0] << ","
      << dim[1] << ","
      << dim[2] << ","
      << ndim << ")" << std::endl;
  }
  // std::cerr << "util::loadTimeSeries: in.size()=" << in.size() << std::endl;


  const float* in_data = reinterpret_cast<const float*> (in.data());
  float* out_data;
  unsigned idx;

  for (unsigned ichan = 0; ichan < dim[0]; ichan++) {
    for (unsigned ipol = 0; ipol < dim[1]; ipol++) {
      out_data = out->get_datptr(ichan, ipol);
      for (unsigned idat = 0; idat < dim[2]; idat++) {
        for (unsigned idim = 0; idim < ndim; idim++) {
          idx = idim + ndim*idat;
          out_data[idx] = in_data[idx];
        }
      }
      in_data += ndim*dim[2];
    }
  }
}



template<typename FilterbankType>
void util::IntegrationTestConfiguration<FilterbankType>::setup (
  dsp::TimeSeries* in,
  dsp::TimeSeries* out
)
{
  if (util::config::verbose) {
    std::cerr << "util::IntegrationTestConfiguration::setup: input_ndat=" << input_ndat << std::endl;
    std::cerr << "util::IntegrationTestConfiguration::setup: input_overlap=" << input_overlap << std::endl;
  }

  auto os_in2out = [this] (unsigned n) -> unsigned {
    return this->os_factor.normalize(n) * this->input_nchan / this->output_nchan;
  };
  unsigned input_fft_length = input_ndat;
  unsigned output_fft_length = os_in2out(input_fft_length);
  unsigned output_overlap = os_in2out(input_overlap);

  std::vector<unsigned> in_dim = {
    input_nchan, npol, input_fft_length*npart};
  std::vector<unsigned> out_dim = {
    output_nchan, npol, output_fft_length*npart};

  unsigned in_size = util::product(in_dim);
  unsigned out_size = util::product(out_dim);

  std::vector<std::complex<float>> in_vec(in_size);
  std::vector<std::complex<float>> out_vec(out_size);

  auto random_gen = util::random<float>();

  for (unsigned idx=0; idx<in_size; idx++) {
    // in_vec[idx] = std::complex<float>((float) idx, (float) idx);
    in_vec[idx] = std::complex<float>(random_gen(), random_gen());
    // std::cerr << in_vec[idx] << std::endl;
  }

  util::loadTimeSeries<std::complex<float>>(in_vec, in, in_dim);
  util::loadTimeSeries<std::complex<float>>(out_vec, out, out_dim);

  filterbank->set_input(in);
	filterbank->set_output(out);

  filterbank->set_oversampling_factor(os_factor);
  filterbank->set_input_fft_length(input_fft_length);
  filterbank->set_output_fft_length(output_fft_length);
  filterbank->set_input_discard_pos(input_overlap);
  filterbank->set_input_discard_neg(input_overlap);
  filterbank->set_output_discard_pos(output_overlap);
  filterbank->set_output_discard_neg(output_overlap);

  overlap_discard_scratch_space = 2*npol*input_nchan*input_fft_length;
  stitching_scratch_space = 2*npol*output_nchan*output_fft_length;

  total_scratch_needed = overlap_discard_scratch_space + stitching_scratch_space;
}


template<typename FilterbankType>
template<typename MemoryType>
std::vector<float*> util::IntegrationTestConfiguration<FilterbankType>::allocate_scratch (
  MemoryType* _memory
)
{
  // if (util::config::verbose) {
  // std::cerr << "util::IntegrationTestConfiguration::allocate_scratch(" << _memory << ")" << std::endl;
  // }

  if (_memory != nullptr) {
    // std::cerr << "util::IntegrationTestConfiguration::allocate_scratch: creating new scratch" << std::endl;
    Reference::To<dsp::Scratch> new_scratch = new dsp::Scratch;
    new_scratch->set_memory(_memory);
    scratch = new_scratch;
  }

  float* space = scratch->space<float> (total_scratch_needed);
  std::vector<float*> space_vector = {space, space + overlap_discard_scratch_space};
  return space_vector;
}






#endif
