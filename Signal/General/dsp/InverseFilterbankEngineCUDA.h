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

#include <cufft.h>

#include "dsp/InverseFilterbankEngine.h"
#include "dsp/LaunchConfig.h"


namespace CUDA
{
  class elapsed
  {
  public:
    elapsed ();
    void wrt (cudaEvent_t before);

    double total;
    cudaEvent_t after;
  };

  //! FFT based PFB inversion implemented using CUDA streams.
  class InverseFilterbankEngineCUDA : public dsp::InverseFilterbank::Engine
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

    //! Setup the Engine's FFT plans. Returns the new scaling factor that will
    //! correctly weight the result of the backward FFT used in `perform`
    double setup_fft_plans (dsp::InverseFilterbank*);

    //! Setup scratch space used in the `perform` member function.
    void set_scratch (float *);

    //! Implements FFT based PFB inversion algorithm using the GPU.
    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);

    //! Get the scaling factor that will correctly scale the result of the
    //! backward FFT used in `perform`.
    double get_scalefac() const {return scalefac;}

    //! Do any actions to clean up after `perform`.
    void finish ();

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
      std::vector<std::complex<float>>& in,
      std::vector<std::complex<float>>& apodization,
      std::vector<std::complex<float>>& out,
      int discard,
      int ndat,
      int nchan);

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
      std::vector<std::complex<float>>& in,
      std::vector<std::complex<float>>& response,
      std::vector<std::complex<float>>& out,
      Rational os_factor,
      int npart,
      int npol,
      int nchan,
      int ndat,
      bool pfb_dc_chan,
      bool pfb_all_chan);

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
      std::vector<std::complex<float>>& in,
      dim3 in_dim,
      std::vector<std::complex<float>>& out,
      dim3 out_dim,
      int discard
    );

  protected:

    //! forward fft plan
    cufftHandle forward;

    //! backward fft plan
    cufftHandle backward;

    //! The type of the forward FFT. The backward plan is always complex to complex.
    cufftType type_forward;

    //! Complex-valued data
    bool real_to_complex;

    //! inplace FFT in CUDA memory
    float2* d_fft;

    //! response or response product in CUDA memory
    float2* d_kernel;

    //! device scratch sapce
    float* scratch;

    LaunchConfig1D multiply;

    cudaStream_t stream;

    bool verbose;

    //! A response object that gets multiplied by assembled spectrum
    dsp::Response* response;

    //! FFT window applied before forward FFT
    dsp::Apodization* fft_window;

  private:
    //! This is the number of floats per sample. This could be 1 or 2,
    //! depending on whether input is Analytic (complex) or Nyquist (real)
    unsigned n_per_sample;

    //! The number of input channels. From the parent dsp::InverseFilterbank
    unsigned input_nchan;
    //! The number of output channels. From the parent dsp::InverseFilterbank
    unsigned output_nchan;

    //! The number of samples discarded at the end of an input TimeSeries. From the parent InverseFilterbank.
    unsigned input_discard_neg;
    //! The number of samples discarded at the start of an input TimeSeries. From the parent InverseFilterbank.
    unsigned input_discard_pos;
    //! The total number of samples discarded in an input TimeSeries. From the parent InverseFilterbank.
    unsigned input_discard_total;

    //! The number of samples discarded at the end of an output TimeSeries. From the parent InverseFilterbank.
    unsigned output_discard_neg;
    //! The number of samples discarded at the start of an output TimeSeries. From the parent InverseFilterbank.
    unsigned output_discard_pos;
    //! The total number of samples discarded ain an input TimeSeries. From the parent InverseFilterbank.
    unsigned output_discard_total;

    //! The number of floats in the forward FFT
    unsigned input_fft_length;
    //! The number of floats in the backward FFT
    unsigned output_fft_length;

    //! The number samples in an input TimeSeries step, or segment. From the parent InverseFilterbank
    unsigned input_sample_step;

    //! The number samples in an output TimeSeries step, or segment. From the parent InverseFilterbank
    unsigned output_sample_step;

    //! How much of the forward FFT to keep due to oversampling
    unsigned input_os_keep;
    //! How much of the forward FFT to discard due to oversampling
    unsigned input_os_discard;

    //! Scratch space for performing forward FFTs
    float* input_fft_scratch;

    //! Scratch space for input time series chunk
    float* input_time_scratch;

    //! Scratch space for performing backward FFTs
    float* output_fft_scratch;

    // float* response_stitch_scratch;
    // float* fft_shift_scratch;

    //! Scratch space where results of forward FFTs get assembled into
    //! upsampled spectrum
    float* stitch_scratch;

    //! Flag indicating whether FFT plans have been setup
    bool fft_plans_setup;

    //! This flag indicates whether we have the DC, or zeroth PFB channel.
    //! From the parent InverseFilterbank
    bool pfb_dc_chan;

    //! This flag indicates whether we have all the channels from the last
    //! stage of upstream channelization.
    //! From the parent InverseFilterbank
    bool pfb_all_chan;


  };

}

#endif
