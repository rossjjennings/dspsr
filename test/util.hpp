#ifndef __util_hpp
#define __util_hpp

#include <fstream>
#include <string>
#include <vector>
#include <complex>

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "Rational.h"

namespace util {

  template<typename T>
  bool isclose (T a, T b, float atol=1e-7, float rtol=1e-5);

  template<typename T>
  bool allclose (T* a, T* b, int size, float atol=1e-7, float rtol=1e-5);

  template<typename T>
  bool allclose (std::vector<T> a, std::vector<T> b, float atol=1e-7, float rtol=1e-5);

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

  template<typename T>
  void response_stitch_cpu (
    std::vector<T> in,
    std::vector<T> resp,
    std::vector<T>& out,
    Rational os_factor,
    int npart,
    int npol,
    int nchan,
    int ndat,
    bool pfb_dc_chan,
    bool pfb_all_chan
  );


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
bool util::allclose (std::vector<T> a, std::vector<T> b, float atol, float rtol) {
  if (a.size() != b.size()) {
     return false;
  }
  return util::allclose(a.data(), b.data(), (int) a.size(), atol, rtol);
}


template<typename T>
bool util::allclose (T* a, T* b, int size, float atol, float rtol)
{
  bool ret = true;
  for (int i=0; i<size; i++) {
    ret = util::isclose<T>(a[i], b[i], atol, rtol);
  }
  return ret;
}

template<typename T>
bool util::isclose (T a, T b, float atol, float rtol)
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

template<typename T>
void util::response_stitch_cpu (
  std::vector<T> in,
  std::vector<T> resp,
  std::vector<T>& out,
  Rational os_factor,
  int npart,
  int npol,
  int nchan,
  int ndat,
  bool pfb_dc_chan,
  bool pfb_all_chan
)
{
  int in_ndat = ndat;
  int in_ndat_keep = os_factor.normalize(in_ndat);
  int in_ndat_keep_2 = in_ndat_keep / 2;
  int out_ndat = nchan * in_ndat_keep;
  int out_size = npart * out_ndat * npol;
  int in_size = npart * npol * nchan * in_ndat;

  if (out.size() != out_size) {
    out.resize(out_size);
  }

  T* in_ptr;
  T* out_ptr;

  int in_offset;
  int out_offset;

  int in_idx_bot;
  int in_idx_top;

  int out_idx_bot;
  int out_idx_top;


  for (int ipart=0; ipart < npart; ipart++) {
    for (int ipol=0; ipol < npol; ipol++) {
      for (int ichan=0; ichan < nchan; ichan++) {
        in_offset = ipart*npol*in_ndat*nchan + ipol*in_ndat*nchan + ichan*in_ndat;
        out_offset = ipart*npol*out_ndat + ipol*out_ndat;
        // std::cerr << "in_offset=" << in_offset << ", out_offset=" << out_offset << std::endl;
        in_ptr = in.data() + in_offset;
        out_ptr = out.data() + out_offset;

        for (int idat=0; idat<in_ndat_keep_2; idat++) {
          in_idx_top = idat;
          in_idx_bot = in_idx_top + (in_ndat - in_ndat_keep_2);

          out_idx_bot = idat + in_ndat_keep*ichan;
          out_idx_top = out_idx_bot + in_ndat_keep_2;

          if (pfb_dc_chan) {
            if (ichan == 0) {
              out_idx_top = idat;
              out_idx_bot = idat + (out_ndat - in_ndat_keep_2);
            } else {
              out_idx_bot = idat + in_ndat_keep*(ichan-1) + in_ndat_keep_2;
              out_idx_top = out_idx_bot + in_ndat_keep_2;
            }
          }

          // std::cerr << in_offset + in_idx_bot << ", " << in_offset + in_idx_top << std::endl;
          // std::cerr << out_offset + out_idx_bot << ", " << out_offset + out_idx_top << std::endl;
          //
          if (in_offset + in_idx_bot > in_size ||
              out_offset + out_idx_top > out_size ||
              in_offset + in_idx_top > in_size ||
              out_offset + out_idx_bot > out_size) {
            std::cerr << "watch out!" << std::endl;
          }
          // std::cerr << "in=[" << in_idx_bot << "," << in_idx_top << "] out=["
          //   << out_idx_bot << "," << out_idx_top << "]" << std::endl;

          out_ptr[out_idx_bot] = resp[out_idx_bot] * in_ptr[in_idx_bot];
          out_ptr[out_idx_top] = resp[out_idx_top] * in_ptr[in_idx_top];

          if (! pfb_all_chan && pfb_dc_chan && ichan == 0) {
            out_ptr[out_idx_bot] = 0.0;
          }
        }
      }
    }
  }





}


#endif
