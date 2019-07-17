#ifndef __util_hpp
#define __util_hpp

#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <chrono>

#include "dsp/IOManager.h"
#include "dsp/Input.h"
#include "dsp/TimeSeries.h"
#include "Rational.h"

namespace util {

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
  bool allclose (T* a, T* b, unsigned size, float atol=1e-7, float rtol=1e-5);

  template<typename T>
  bool allclose (std::vector<T> a, std::vector<T> b, float atol=1e-7, float rtol=1e-5);

  template<typename T>
  void load_binary_data (std::string file_path, std::vector<T>& test_data);

  template<typename T>
  void write_binary_data (std::string file_path, std::vector<T>& data);

  template<typename T>
  void write_binary_data (std::string file_path, T* buffer, unsigned len);

  template<typename T>
  void print_array (std::vector<T>& arr, std::vector<unsigned>& dim);

  template<typename T>
  void print_array (T* arr, std::vector<unsigned>& dim);

  void load_psr_data (dsp::IOManager manager, unsigned block_size, dsp::TimeSeries* ts);

  void set_verbose (bool val);

  std::string get_test_data_dir ();

  //! load data from a stl vector unsignedo a dspsr TimeSeries object.
  //! dim should be of length 3.
  //! Assumes TimeSeries is in FPT order.
  template<typename T>
  void loadTimeSeries (
    const std::vector<T>& in,
    dsp::TimeSeries* out,
    const std::vector<unsigned>& dim);

  template<typename T>
  void response_stitch_cpu (
    std::vector<T> in,
    std::vector<T> resp,
    std::vector<T>& out,
    Rational os_factor,
    unsigned npart,
    unsigned npol,
    unsigned nchan,
    unsigned ndat,
    bool pfb_dc_chan,
    bool pfb_all_chan
  );

  template<typename T>
  void apodization_overlap_cpu (
    std::vector<T> in,
    std::vector<T> apodization,
    std::vector<T>& out,
    unsigned discard,
    unsigned npart,
    unsigned npol,
    unsigned nchan,
    unsigned ndat
  );

  template<typename T>
  void overlap_discard_cpu (
    std::vector<T> in,
    std::vector<T>& out,
    unsigned discard,
    unsigned npart,
    unsigned npol,
    unsigned nchan,
    unsigned ndat
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
bool util::allclose (std::vector<T> a, std::vector<T> b, float atol, float rtol) {
  if (a.size() != b.size()) {
     return false;
  }
  return util::allclose(a.data(), b.data(), (unsigned) a.size(), atol, rtol);
}


template<typename T>
bool util::allclose (T* a, T* b, unsigned size, float atol, float rtol)
{
  bool ret = true;
  for (unsigned i=0; i<size; i++) {
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
void util::print_array (std::vector<T>& arr, std::vector<unsigned>& dim)
{
  util::print_array<T>(arr.data(), dim);
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

  std::cerr << "util::loadTimeSeries: ("
    << dim[0] << ","
    << dim[1] << ","
    << dim[2] << ","
    << ndim << ")" << std::endl;
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




template<typename T>
void util::response_stitch_cpu (
  std::vector<T> in,
  std::vector<T> resp,
  std::vector<T>& out,
  Rational os_factor,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned ndat,
  bool pfb_dc_chan,
  bool pfb_all_chan
)
{
  unsigned in_ndat = ndat;
  unsigned in_ndat_keep = os_factor.normalize(in_ndat);
  unsigned in_ndat_keep_2 = in_ndat_keep / 2;
  unsigned out_ndat = nchan * in_ndat_keep;
  unsigned out_size = npart * out_ndat * npol;
  unsigned in_size = npart * npol * nchan * in_ndat;

  if (out.size() != out_size) {
    out.resize(out_size);
  }

  T* in_ptr;
  T* out_ptr;

  unsigned in_offset;
  unsigned out_offset;

  unsigned in_idx_bot;
  unsigned in_idx_top;

  unsigned out_idx_bot;
  unsigned out_idx_top;


  for (unsigned ipart=0; ipart < npart; ipart++) {
    for (unsigned ipol=0; ipol < npol; ipol++) {
      for (unsigned ichan=0; ichan < nchan; ichan++) {
        in_offset = ipart*npol*in_ndat*nchan + ipol*in_ndat*nchan + ichan*in_ndat;
        out_offset = ipart*npol*out_ndat + ipol*out_ndat;
        // std::cerr << "in_offset=" << in_offset << ", out_offset=" << out_offset << std::endl;
        in_ptr = in.data() + in_offset;
        out_ptr = out.data() + out_offset;

        for (unsigned idat=0; idat<in_ndat_keep_2; idat++) {
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

template<typename T>
void util::apodization_overlap_cpu (
  std::vector<T> in,
  std::vector<T> apodization,
  std::vector<T>& out,
  unsigned discard,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned ndat
)
{
  unsigned out_ndat = ndat - 2*discard;

  unsigned in_size = npart * npol * nchan * ndat;
  unsigned out_size = npart * npol * nchan * out_ndat;
  unsigned apod_size = nchan * out_ndat;

  unsigned in_offset;
  unsigned out_offset;
  unsigned apod_offset;

  for (unsigned ipart=0; ipart<npart; ipart++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      for (unsigned ichan=0; ichan<nchan; ichan++) {
        in_offset = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat;
        out_offset = ipart*npol*nchan*out_ndat + ipol*nchan*out_ndat + ichan*out_ndat;
        apod_offset = ichan*out_ndat;
        for (unsigned idat=0; idat<out_ndat; idat++) {

          if (
            idat + out_offset > out_size ||
            apod_offset + idat > apod_size ||
            idat + discard + in_offset > in_size
          ) {
            std::cerr << "util::apodization_overlap_cpu: watch out!" << std::endl;
          }

          if (apodization.size() != 0) {
            out[idat + out_offset] = apodization[apod_offset + idat]*in[idat + discard + in_offset];
          } else {
            out[idat + out_offset] = in[idat + discard + in_offset];
          }
        }
      }
    }
  }
}

template<typename T>
void util::overlap_discard_cpu (
  std::vector<T> in,
  std::vector<T>& out,
  unsigned discard,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned ndat
)
{
  std::vector<T> empty;
  util::apodization_overlap_cpu<T>(
    in, empty, out, discard, npart, npol, nchan, ndat
  );
}


#endif
