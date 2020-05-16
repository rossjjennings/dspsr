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
#include <functional>

// #include "json.hpp"
#include "toml.h"

#include "config.h"
#include "dsp/Apodization.h"
#include "dsp/Response.h"
#include "dsp/Scratch.h"
#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "Rational.h"

#if HAVE_CUDA
#include "dsp/MemoryCUDA.h"
#include "dsp/TransferCUDA.h"
#endif

// using json = nlohmann::json;

void check_error (const char*);

namespace test {
namespace util {

  // template<typename ReturnType, typename ClsType, typename ... Args >
  // struct lambda_details
  // {
  //   using return_type = ReturnType;
  //   using arity = std::integral_constant<std::size_t, sizeof...(Args)>;
  //
  // };
  //
  // template<typename FunctorType>
  // struct lambda_traits : lambda_traits<decltype(&FunctorType::operator())> {};
  //
  // template<typename ReturnType, typename ClsType, typename ... Args>
  // struct lambda_traits<ReturnType(ClsType::*)(Args...) const> : lambda_details<ReturnType, ClsType, Args...> {};


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

  void to_toml (toml::Value& val, const TestShape& sh);

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

  template<typename T>
  struct tomltype;

  template<>
  struct tomltype<float> {
    typedef double type;
  };

  template<>
  struct tomltype<double> {
    typedef double type;
  };

  template<>
  struct tomltype<unsigned> {
    typedef int type;
  };

  template<>
  struct tomltype<int> {
    typedef int type;
  };

  template<>
  struct tomltype<bool> {
    typedef bool type;
  };

  template<>
  struct tomltype<std::string> {
    typedef std::string type;
  };

  template<typename unit>
  using delta = std::chrono::duration<double, unit>;

  typedef std::chrono::time_point<std::chrono::high_resolution_clock> time_point;

  time_point now ();

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
  T max (const T* a, unsigned size);

  template<typename T>
  T max (const std::vector<T>& a);

  template<typename T>
  T min (const T* a, unsigned size);

  template<typename T>
  T min (const std::vector<T>& a);

  template<typename T>
  std::vector<T> subtract (const std::vector<T>& a, const std::vector<T>& b);

  template<typename T>
  void subtract (T* a, T* b, T* res, unsigned size);

  template<typename T>
  std::vector<T> abs (const std::vector<T>& a);

  template<typename T>
  void abs (T* a, unsigned size);

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

  void load_psr_data (
    dsp::IOManager manager,
    unsigned block_size,
    dsp::TimeSeries* ts,
    int nblocks=-1);

  void set_verbose (bool val);

  std::string get_test_env_var (const std::string& env_var_name,const std::string& default_val = ".");

  std::string get_test_data_dir ();

  std::string get_working_path ();
  //! load data from a stl vector to a dspsr TimeSeries object.
  //! dim should be of length 3.
  //! Assumes TimeSeries is in FPT order.
  template<typename T>
  void loadTimeSeries (
    const std::vector<T>& in,
    dsp::TimeSeries* out,
    const std::vector<unsigned>& dim);

  template<typename FunctionType>
  void init_TimeSeries(dsp::TimeSeries* ts, FunctionType filler);

  template<typename FunctionType>
  uint64_t check_TimeSeries(dsp::TimeSeries* ts, FunctionType checker);

  std::function<void(dsp::TimeSeries*, dsp::TimeSeries*, cudaMemcpyKind)> transferTimeSeries (
    cudaStream_t stream, CUDA::DeviceMemory* memory);

}
}

template<typename T>
void test::util::load_binary_data (std::string file_path, std::vector<T>& test_data)
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
void test::util::write_binary_data (std::string file_path, std::vector<T> buffer)
{
  test::util::write_binary_data(file_path, buffer.data(), buffer.size());
}

template<typename T>
void test::util::write_binary_data (std::string file_path, T* buffer, unsigned len)
{
  std::ofstream file(file_path, std::ios::out | std::ios::binary);

  file.write(
    reinterpret_cast<const char*>(buffer),
    len*sizeof(T)
  );
  file.close();
}


template<typename T>
unsigned test::util::nclose (const std::vector<T>& a, const std::vector<T>& b, float atol, float rtol) {

  if (a.size() != b.size()) {
    std::string msg ("test::util::nclose: vectors not the same size");
    if (test::util::config::verbose) {
     std::cerr << msg << std::endl;
    }
    throw msg;
  }
  if (test::util::config::verbose) {
    std::cerr << "test::util::nclose size=" << a.size() << std::endl;
  }
  return test::util::nclose(a.data(), b.data(), (unsigned) a.size(), atol, rtol);
}


template<typename T>
unsigned test::util::nclose (const T* a, const T* b, unsigned size, float atol, float rtol)
{
  unsigned nclose=0;
  bool close;
  for (unsigned i=0; i<size; i++) {
    close = test::util::isclose<T>(a[i], b[i], atol, rtol);
    if (close) {
      nclose++;
    }
  }
  return nclose;
}


template<typename T>
bool test::util::allclose (const std::vector<T>& a, const std::vector<T>& b, float atol, float rtol) {

  return test::util::nclose<T> (a, b, atol, rtol) == a.size();
}


template<typename T>
bool test::util::allclose (const T* a, const T* b, unsigned size, float atol, float rtol)
{
  return test::util::nclose<T> (a, b, size, atol, rtol) == size;
}

template<typename T>
bool test::util::isclose (T a, T b, float atol, float rtol)
{
  return std::abs(a - b) <= (atol + rtol*std::abs(b));
}

template<typename T>
T test::util::max (const T* a, unsigned size)
{
  T max_val = a[0];
  for (unsigned idx=1; idx<size; idx++)
  {
    if (a[idx] > max_val) {
      max_val = a[idx];
    }
  }
  return max_val;
}

template<typename T>
T test::util::max (const std::vector<T>& a)
{
  return test::util::max<T>(a.data(), a.size());
}

template<typename T>
T test::util::min (const T* a, unsigned size)
{
  T min_val = a[0];
  for (unsigned idx=1; idx<size; idx++)
  {
    if (a[idx] < min_val) {
      min_val = a[idx];
    }
  }
  return min_val;
}

template<typename T>
T test::util::min (const std::vector<T>& a)
{
  return test::util::min<T>(a.data(), a.size());
}


template<typename T>
std::vector<T> test::util::subtract (const std::vector<T>& a, const std::vector<T>& b)
{
}

template<typename T>
void test::util::subtract (const T* a, const T* b, T* res, unsigned size)
{
}

template<typename T>
std::vector<T> test::util::abs (const std::vector<T>& a)
{
  std::vector<T> res(a.size());
  test::util::abs (a.data(), res.data(), a.size());
}

template<typename T>
void test::util::abs (const T* a, T* res, unsigned size)
{
}



template<typename T>
void test::util::print_array (const std::vector<T>& arr, std::vector<unsigned>& dim)
{
  test::util::print_array<T>(const_cast<T*>(arr.data()), dim);
}

template<typename T>
void test::util::print_array (T* arr, std::vector<unsigned>& dim)
{
  if (dim.size() > 2) {
    unsigned head_dim = dim[0];
    std::vector<unsigned> tail_dim(dim.begin() + 1, dim.end());
    unsigned stride = 1;
    for (unsigned d=1; d<dim.size(); d++) {
      stride *= dim[d];
    }
    for (unsigned i=0; i<head_dim; i++) {
      test::util::print_array<T>(arr + stride*i, tail_dim);
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
T test::util::sum (std::vector<T> a)
{
  T res = 0;
  std::for_each (a.begin(), a.end(), [&res](unsigned i){res+=i;});
  return res;
}

template<typename T>
T test::util::product (std::vector<T> a)
{
  T res = 1;
  std::for_each (a.begin(), a.end(), [&res](unsigned i){res*=i;});
  return res;
}

template<typename T>
std::function<T(void)> test::util::random ()
{
  srand (time(NULL));
  return [] () { return (T) rand() / RAND_MAX; };
}


template<typename T>
void test::util::loadTimeSeries (
  const std::vector<T>& in,
  dsp::TimeSeries* out,
  const std::vector<unsigned>& dim
)
{
  typedef test::util::std2dspsr<T> dspsr_type;

  unsigned ndim = dspsr_type::ndim;

  out->set_state (dspsr_type::state);

  out->set_nchan (dim[0]);
  out->set_npol (dim[1]);
  out->set_ndat (dim[2]);
  out->set_ndim (ndim);
  out->resize (dim[2]);

  if (test::util::config::verbose) {
    std::cerr << "test::util::loadTimeSeries: ("
      << dim[0] << ","
      << dim[1] << ","
      << dim[2] << ","
      << ndim << ")" << std::endl;
  }
  // std::cerr << "test::util::loadTimeSeries: in.size()=" << in.size() << std::endl;


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

template<typename FunctionType>
void test::util::init_TimeSeries (
  dsp::TimeSeries* ts,
  FunctionType filler
)
{
  float* ts_data;
  unsigned ndim = ts->get_ndim();
  unsigned idx = 0;
  for (unsigned ichan=0; ichan<ts->get_nchan(); ichan++) {
    for (unsigned ipol=0; ipol < ts->get_npol(); ipol++) {
      ts_data = ts->get_datptr(ichan, ipol);
      for (unsigned idat = 0; idat < ts->get_ndat(); idat++) {
        for (unsigned idim = 0; idim < ndim; idim++) {
          idx = idim + ndim*idat;
          ts_data[idx] = filler(ichan, ipol, idat, idim);
        }
      }
    }
  }
}

template<typename FunctionType>
uint64_t test::util::check_TimeSeries (
  dsp::TimeSeries* ts,
  FunctionType checker
)
{
  uint64_t error_count = 0;
  float* ts_data;
  unsigned ndim = ts->get_ndim();
  unsigned idx = 0;
  for (unsigned ichan=0; ichan<ts->get_nchan(); ichan++) {
    for (unsigned ipol=0; ipol < ts->get_npol(); ipol++) {
      ts_data = ts->get_datptr(ichan, ipol);
      for (unsigned idat = 0; idat < ts->get_ndat(); idat++) {
        for (unsigned idim = 0; idim < ndim; idim++) {
          idx = idim + ndim*idat;
          if (!isclose(ts_data[idx], checker(ichan, ipol, idat, idim)))
            error_count++;
        }
      }
    }
  }
  return error_count;
}

#endif
