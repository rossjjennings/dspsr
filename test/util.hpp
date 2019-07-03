#ifndef __util_hpp
#define __util_hpp

#include <fstream>
#include <string>
#include <vector>
#include <complex>

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"

namespace util {

  template<typename T>
  bool isclose (T a, T b, T atol, T rtol);

  template<typename T>
  bool compare_test_data (T* a, T* b, int size, T atol=1e-7, T rtol=1e-5);

  template<typename T>
  void load_binary_data (std::string file_path, std::vector<T>& test_data);

  template<typename T>
  void write_binary_data (std::string file_path, std::vector<T>& data);

  template<typename T>
  void write_binary_data (std::string file_path, T* buffer, int len);

  template<typename T>
  void print_array (std::vector<T>& arr, std::vector<int>& dim);

  template<typename T>
  void print_array (T* arr, std::vector<int>& dim);

  void load_psr_data (dsp::IOManager manager, int block_size, dsp::TimeSeries* ts);

  void set_verbose (bool val);

  std::string get_test_data_dir ();

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

    int T_size = (size / sizeof(T));
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
void util::write_binary_data (std::string file_path, T* buffer, int len)
{
  std::ofstream file(file_path, std::ios::out | std::ios::binary);

  file.write(
    reinterpret_cast<const char*>(buffer),
    len*sizeof(T)
  );
  file.close();
}

template<typename T>
bool util::compare_test_data (T* a, T* b, int size, T atol, T rtol)
{
  bool ret = true;
  bool val;
  for (int i=0; i<size; i++) {
    val = util::isclose(a[i], b[i], atol, rtol);
    if (! val) {
      ret = false;
      std::cerr << "i=" << i << " ("<< a[i] << ", " << b[i] << ")" << std::endl;
    }
  }
  return ret;
}

template<typename T>
bool util::isclose (T a, T b, T atol, T rtol)
{
  return abs(a - b) <= (atol + rtol * abs(b));
}

template<typename T>
void util::print_array (std::vector<T>& arr, std::vector<int>& dim)
{
  util::print_array<T>(arr.data(), dim);
}

template<typename T>
void util::print_array (T* arr, std::vector<int>& dim)
{
  if (dim.size() > 2) {
    int head_dim = dim[0];
    std::vector<int> tail_dim(dim.begin() + 1, dim.end());
    int stride = 1;
    for (int d=1; d<dim.size(); d++) {
      stride *= dim[d];
    }
    for (int i=0; i<head_dim; i++) {
      util::print_array<T>(arr + stride*i, tail_dim);
    }
  } else {
    for (int i=0; i<dim[0]; i++) {
      for (int j=0; j<dim[1]; j++) {
        std::cerr << arr[i*dim[1] + j] << " ";
      }
      std::cerr << std::endl;
    }
    std::cerr << std::endl;
  }
}

#endif
