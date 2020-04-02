// configuration for InverseFilterbank testing

#ifndef __InverseFilterbankTestConfig_hpp
#define __InverseFilterbankTestConfig_hpp

#include <vector>

#include "dsp/InverseFilterbank.h"

#include "util/TestConfig.hpp"

namespace test {
namespace util {
  namespace InverseFilterbank {

    class InverseFilterbankProxy {

    public:

      InverseFilterbankProxy (
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
        filterbank = new dsp::InverseFilterbank;
      }

      void setup (
        dsp::TimeSeries* in,
        dsp::TimeSeries* out,
        bool, bool
      );

      float* allocate_scratch(unsigned total_scratch_needed);

      template<class MemoryType>
      void set_memory (MemoryType* _memory);

      Reference::To<dsp::Scratch> scratch;
      Reference::To<dsp::InverseFilterbank> filterbank;

    private:

      const Rational& os_factor;
      unsigned npart;
      unsigned npol;
      unsigned input_nchan;
      unsigned output_nchan;
      unsigned input_ndat;
      unsigned input_overlap;

    };

    class InverseFilterbankTestConfig : public test::util::TestConfig {

    public:

      std::vector<TestShape> get_test_vector_shapes ();

    };

    template<typename T>
    void response_stitch_cpu_FPT (
      const std::vector<T>& in,
      const std::vector<T>& resp,
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
    void apodization_overlap_cpu_FPT (
      const std::vector<T>& in,
      const std::vector<T>& apodization,
      std::vector<T>& out,
      unsigned discard,
      unsigned npart,
      unsigned npol,
      unsigned nchan,
      unsigned samples_per_part,
      unsigned in_ndat,
      unsigned out_ndat
    );

    template<typename T>
    void overlap_discard_cpu_FPT (
      const std::vector<T>& in,
      std::vector<T>& out,
      unsigned discard,
      unsigned npart,
      unsigned npol,
      unsigned nchan,
      unsigned samples_per_part,
      unsigned in_ndat,
      unsigned out_ndat
    );

    template<typename T>
    void overlap_save_cpu_FPT (
      const std::vector<T>& in,
      std::vector<T>& out,
      unsigned discard,
      unsigned npart,
      unsigned npol,
      unsigned nchan,
      unsigned samples_per_part,
      unsigned in_ndat,
      unsigned out_ndat
    );
  }
}
}

template<typename MemoryType>
void test::util::InverseFilterbank::InverseFilterbankProxy::set_memory (
  MemoryType* _memory
)
{
  if (_memory != nullptr) {
    Reference::To<dsp::Scratch> new_scratch = new dsp::Scratch;
    new_scratch->set_memory(_memory);
    scratch = new_scratch;
  }
}



template<typename T>
void test::util::InverseFilterbank::response_stitch_cpu_FPT (
  const std::vector<T>& in,
  const std::vector<T>& resp,
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

  const T* in_ptr;
  T* out_ptr;

  unsigned in_offset;
  unsigned out_offset;

  unsigned in_idx_bot;
  unsigned in_idx_top;

  unsigned out_idx_bot;
  unsigned out_idx_top;


  for (unsigned ichan=0; ichan < nchan; ichan++) {
    for (unsigned ipol=0; ipol < npol; ipol++) {
      for (unsigned ipart=0; ipart < npart; ipart++) {
        in_offset = ichan*npol*npart*in_ndat + ipol*npart*in_ndat + ipart*in_ndat;
        out_offset = ipol*npart*out_ndat + ipart*out_ndat;
        // in_offset = ipart*npol*in_ndat*nchan + ipol*in_ndat*nchan + ichan*in_ndat;
        // out_offset = ipart*npol*out_ndat + ipol*out_ndat;
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
void test::util::InverseFilterbank::apodization_overlap_cpu_FPT (
  const std::vector<T>& in,
  const std::vector<T>& apodization,
  std::vector<T>& out,
  unsigned discard,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned samples_per_part,
  unsigned in_ndat,
  unsigned out_ndat
)
{
  unsigned total_discard = 2*discard;
  unsigned step = samples_per_part - total_discard;

  unsigned in_size = npol * nchan * in_ndat;
  unsigned out_size = npol * nchan * out_ndat;
  unsigned apod_size = samples_per_part;

  unsigned in_offset;
  unsigned out_offset;

  // order is FPT
  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      for (unsigned ipart=0; ipart<npart; ipart++) {
        in_offset = ichan*npol*in_ndat + ipol*in_ndat + ipart*step;
        out_offset = ichan*npol*out_ndat + ipol*out_ndat + ipart*samples_per_part;
        for (unsigned idat=0; idat<samples_per_part; idat++) {
          if (
            idat + out_offset > out_size ||
            idat > apod_size ||
            idat + in_offset > in_size
          ) {
            std::cerr << "test::util::apodization_overlap_cpu_FPT: watch out!" << std::endl;
            std::cerr << "test::util::apodization_overlap_cpu_FPT: idat + out_offset=" << idat + out_offset << ", out_size=" << out_size << std::endl;
            std::cerr << "test::util::apodization_overlap_cpu_FPT: idat + in_offset=" << idat + in_offset << ", in_size=" << in_size << std::endl;
            std::cerr << "test::util::apodization_overlap_cpu_FPT: idat=" << idat  << ", apod_size=" << apod_size << std::endl;
          }

          if (apodization.size() != 0) {
            out[out_offset + idat] = apodization[idat]*in[in_offset + idat];
          } else {
            out[out_offset + idat] = in[in_offset + idat];
          }
        }
      }
    }
  }
}

template<typename T>
void test::util::InverseFilterbank::overlap_discard_cpu_FPT (
  const std::vector<T>& in,
  std::vector<T>& out,
  unsigned discard,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned samples_per_part,
  unsigned in_ndat,
  unsigned out_ndat
)
{
  std::vector<T> empty;
  test::util::InverseFilterbank::apodization_overlap_cpu_FPT<T>(
    in, empty, out, discard, npart, npol, nchan, samples_per_part, in_ndat, out_ndat
  );
}

template<typename T>
void test::util::InverseFilterbank::overlap_save_cpu_FPT (
  const std::vector<T>& in,
  std::vector<T>& out,
  unsigned discard,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned samples_per_part,
  unsigned in_ndat,
  unsigned out_ndat
)
{
  unsigned total_discard = 2*discard;
  unsigned step = samples_per_part - total_discard;

  unsigned in_size = npol * nchan * in_ndat;
  unsigned out_size = npol * nchan * out_ndat;

  unsigned in_offset;
  unsigned out_offset;

  // order is FPT
  for (unsigned ichan=0; ichan<nchan; ichan++) {
    for (unsigned ipol=0; ipol<npol; ipol++) {
      for (unsigned ipart=0; ipart<npart; ipart++) {
        in_offset = ichan*npol*in_ndat + ipol*in_ndat + ipart*samples_per_part;
        out_offset = ichan*npol*out_ndat + ipol*out_ndat + ipart*step;
        for (unsigned idat=0; idat<step; idat++) {
          if (
            idat + out_offset > out_size ||
            idat + in_offset > in_size
          ) {
            std::cerr << "test::util::overlap_save_cpu_FPT: watch out!" << std::endl;
            std::cerr << "test::util::overlap_save_cpu_FPT: idat + out_offset=" << idat + out_offset << ", out_size=" << out_size << std::endl;
            std::cerr << "test::util::overlap_save_cpu_FPT: idat + in_offset=" << idat + in_offset << ", in_size=" << in_size << std::endl;
          }
          out[out_offset + idat] = in[in_offset + idat + discard];
        }
      }
    }
  }
}

#endif
