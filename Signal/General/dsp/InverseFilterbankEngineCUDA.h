//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FilterbankCUDA.h

#ifndef __InverseFilterbankEngineCUDACUDA_h
#define __InverseFilterbankEngineCUDACUDA_h

#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/LaunchConfig.h"

#include <cufft.h>

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

  //! Discrete convolution filterbank step implemented using CUDA streams
  class InverseFilterbankEngineCUDA : public dsp::InverseFilterbank::Engine
  {
    unsigned nstream;

  public:

    //! Default Constructor
    InverseFilterbankEngineCUDA (cudaStream_t stream);

    ~InverseFilterbankEngineCUDA ();

    //! Use the parent `InverseFilterbank` object to set properties used in the
    //! `perform` member function
    void setup (InverseFilterbank*);

    //! Setup the Engine's FFT plans. Returns the new scaling factor that will
    //! correctly weight the result of the backward FFT used in `perform`
    double setup_fft_plans (InverseFilterbank*);

    //! Setup scratch space used in the `perform` member function.
    void set_scratch (float *);

    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);

    //! Get the scaling factor that will correctly scale the result of the
    //! backward FFT used in `perform`
    double get_scalefac() const {return scalefac;}

    void finish ();

  protected:

    //! forward fft plan
    cufftHandle forward;

    //! backward fft plan
    cufftHandle backward;

    //! Complex-valued data
    bool real_to_complex;

    //! inplace FFT in CUDA memory
    float2* d_fft;

    //! convolution kernel in CUDA memory
    float2* d_kernel;

    //! device scratch sapce
    float* scratch;

    LaunchConfig1D multiply;

    cudaStream_t stream;

    bool verbose;

    //! A response object that gets multiplied by assembled spectrum
    Response* response;

    //! FFT window applied before forward FFT
    Apodization* fft_window;

  private:
    //! This is the number of floats per sample. This could be 1 or 2,
    //! depending on whether input is Analytic (complex) or Nyquist (real)
    unsigned n_per_sample;

    //! The number of input channels. From the parent InverseFilterbank
    unsigned input_nchan;
    //! The number of output channels. From the parent InverseFilterbank
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
