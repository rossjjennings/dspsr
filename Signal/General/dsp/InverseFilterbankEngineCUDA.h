//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FilterbankCUDA.h

#ifndef __InverseFilterbankEngineCUDA_h
#define __InverseFilterbankEngineCUDA_h

#include <complex>
#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>

#include "dsp/InverseFilterbankEngineCPU.h"
#include "dsp/LaunchConfig.h"


namespace CUDA
{

  template<int i>
  struct cufftTypeMap;

  template<>
  struct cufftTypeMap<CUFFT_R2C> {
    static const cufftType type = CUFFT_R2C;
    typedef cufftReal input_type;
    typedef cufftComplex output_type;
    typedef float input_std_type;
    typedef std::complex<float> output_std_type;
    static cufftResult cufftExec (cufftHandle plan, input_type* in, output_type* out, int flag) {
        return cufftExecR2C(plan, in, out);
    };
  };

  template<>
  struct cufftTypeMap<CUFFT_C2C> {
    static const cufftType type = CUFFT_C2C;
    typedef cufftComplex input_type;
    typedef cufftComplex output_type;
    typedef std::complex<float> input_std_type;
    typedef std::complex<float> output_std_type;
    static cufftResult cufftExec (cufftHandle plan, input_type* in, output_type* out, int flag) {
        return cufftExecC2C(plan, in, out, flag);
    };
  };



  class elapsed
  {
  public:
    elapsed ();
    void wrt (cudaEvent_t before);

    double total;
    cudaEvent_t after;
  };

  //! FFT based PFB inversion implemented using CUDA streams.
  class InverseFilterbankEngineCUDA : public dsp::InverseFilterbankEngineCPU
  {
    unsigned nstream;

  public:

    //! Default Constructor. This also allocates memory for cuFFT plans
    InverseFilterbankEngineCUDA (cudaStream_t stream);

    //! Default Destructor. This frees up cuFFT plan memory.
    ~InverseFilterbankEngineCUDA ();

    //! Use the parent `InverseFilterbank` object to set properties used in the
    //! `perform` member function
    void setup (dsp::InverseFilterbank*);

    //! Setup scratch space used in the `perform` member function.
    void set_scratch (float *);

    //! Implements FFT based PFB inversion algorithm using the GPU.
    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);

    //! Do any actions to clean up after `perform`.
    void finish ();

    //! setup backward fft plans. This is public so we can test it.
    std::vector<cufftResult> setup_backward_fft_plan (
      int _output_fft_length,
      int _output_nchan
    );

    //! setup forward fft plans. This is public so we can test it.
    std::vector<cufftResult> setup_forward_fft_plan (
      int _input_fft_length,
      int _input_nchan,
      cufftType _type_forward
    );


    template<int i>
    void apply_cufft_forward (
      std::vector< typename cufftTypeMap<i>::input_std_type >& in,
      std::vector< typename cufftTypeMap<i>::output_std_type >& out
    );

    //! Apply the cufft backwards FFT plan to some data.
    void apply_cufft_backward (
      std::vector< std::complex<float> >& in,
      std::vector< std::complex<float> >& out
    );



    //! Apply the k_apodization_overlap kernel to some data.
    //! This function copies arrays from host to device, so it is not intended
    //! to be performant.
    //! \param in input array buffer
    //! \param apodization time domain windowing function, as complex buffer
    //! \param out output array buffer
    //! \param discard the size of the discard region, in complex samples
    //! \param ndat the size of the input array buffer, in complex samples
    //! \param nchan the number of channels in the input array
    static void apply_k_apodization_overlap (
      std::vector< std::complex<float> >& in,
      std::vector< std::complex<float> >& apodization,
      std::vector< std::complex<float> >& out,
      int discard,
      int npart,
      int npol,
      int nchan,
      int ndat
    );

    //! Apply the k_apodization_overlap kernel to some data.
    //! This function copies arrays from host to device, so it is not intended
    //! to be performant.
    //! \param in input array buffer, dimensions (in_nchan*npol, in_ndat)
    //! \param response array buffer, dimensions (out_ndat)
    //! \param out output array buffer, dimensions (npol, out_ndat)
    //! \param os_factor the oversampling factor associated with the
    //!     channelized input data
    //! \param npol the number of polarizations present in the data.
    //! \param in_nchan the number of channels in the input array
    //! \param in_ndat the second dimension in the input array
    //! \param out_ndat the second dimension of the output array
    //! \param pfb_dc_chan whether or not the PFB DC channel is present
    //! \param pfb_all_chan whether or not all the PFB channels are present
    static void apply_k_response_stitch (
      std::vector< std::complex<float> >& in,
      std::vector< std::complex<float> >& response,
      std::vector< std::complex<float> >& out,
      Rational os_factor,
      int npart,
      int npol,
      int nchan,
      int ndat,
      bool pfb_dc_chan,
      bool pfb_all_chan
    );

    //! Apply the k_overlap_discard kernel to some data.
    //! This function copies arrays from host to device, so it is not intended
    //! to be performant.
    //! \param in input array buffer
    //! \param in_dim dimensions of in array
    //! \param out output array buffer
    //! \param out_dim dimensions of out array
    //! \param discard number of samples to discard from either side of each
    //!     channel
    static void apply_k_overlap_discard (
      std::vector< std::complex<float> >& in,
      std::vector< std::complex<float> >& out,
      int discard,
      int npart,
      int npol,
      int nchan,
      int ndat
    );

  private:

    cudaStream_t stream;

    //! forward fft plan
    cufftHandle forward;

    //! backward fft plan
    cufftHandle backward;

    //! The type of the forward FFT. The backward plan is always complex to complex.
    cufftType type_forward;

    //! inplace FFT in CUDA memory
    float2* d_fft;

    //! response or response product in CUDA memory
    float2* d_kernel;
  };
}

template<int i>
void CUDA::InverseFilterbankEngineCUDA::apply_cufft_forward (
  std::vector< typename CUDA::cufftTypeMap<i>::input_std_type >& in,
  std::vector< typename CUDA::cufftTypeMap<i>::output_std_type >& out
)
{

}




#endif
