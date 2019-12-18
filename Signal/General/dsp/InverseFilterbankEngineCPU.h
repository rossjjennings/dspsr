//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/InverseFilterbankEngineCPU.h

#ifndef __InverseFilterbankEngineCPU_h
#define __InverseFilterbankEngineCPU_h

#include "dsp/InverseFilterbankEngine.h"

#include "FTransform.h"

namespace dsp
{

  //! InverseFilterbankEngineCPU is the `InverseFilterbank` engine that
  //! runs on the CPU. This class implements the PFB inversion algorithm,
  //! which synthesizes some channelized input into a lower (single) number of
  //! output channels at a higher sampling rate.
  class InverseFilterbankEngineCPU : public dsp::InverseFilterbank::Engine
  {

  public:

    //! Default Constructor
    InverseFilterbankEngineCPU ();

    ~InverseFilterbankEngineCPU ();

    //! Use the parent `InverseFilterbank` object to set properties used in the
    //! `perform` member function
    void setup (InverseFilterbank*);

    //! Setup scratch space used in the `perform` member function.
    void set_scratch (float *);

    //! Operate on input and output data TimeSeries, performing the PFB
    //! inversion algorithm.
    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step=0, uint64_t out_step=0);

    void perform (const dsp::TimeSeries * in,
                  dsp::TimeSeries * out,
                  dsp::TimeSeries* zero_DM_out,
                  uint64_t npart,
                  const uint64_t in_step=0,
                  const uint64_t out_step=0);

    //! Called when the the `InverseFilterbank` sees that the engine is done
    //! operating on data
    void finish ();

  protected:

    //! Complex-valued data
    bool real_to_complex;

    //! verbosity flag
    bool verbose;

    //! A response object that gets multiplied by assembled spectrum
    Response* response;

    //! zero DM response
    Response* zero_DM_response;

    //! FFT window applied before forward FFT
    Apodization* fft_window;

    //! This is the number of floats per sample. This could be 1 or 2,
    //! depending on whether input is Analytic (complex) or Nyquist (real)
    unsigned n_per_sample;

    //! the number of input polarizations
    unsigned input_npol;

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


    //! Flag indicating whether FFT plans have been setup
    bool fft_plans_setup;

    //! This flag indicates whether we have the DC, or zeroth PFB channel.
    //! From the parent InverseFilterbank
    bool pfb_dc_chan;

    //! This flag indicates whether we have all the channels from the last
    //! stage of upstream channelization.
    //! From the parent InverseFilterbank
    bool pfb_all_chan;

  private:


    //! device scratch space
    float* scratch;

    //! plan for computing forward fourier transforms
    FTransform::Plan* forward;

    //! plan for computing inverse fourier transforms
    FTransform::Plan* backward;

    //! Scratch space for performing forward FFTs
    float* input_fft_scratch;

    //! Scratch space, in samples, needed for the input_fft_scratch space
    unsigned input_fft_scratch_samples;

    //! Scratch space for input time series chunk
    float* input_time_scratch;

    //! Scratch space, in samples, needed for the input_time_scratch space
    unsigned input_time_scratch_samples;

    //! Scratch space for performing backward FFTs
    float* output_fft_scratch;

    //! Scratch space, in samples, needed for the output_fft_scratch space
    unsigned output_fft_scratch_samples;

    //! Scratch space where results of forward FFTs get assembled into
    //! upsampled spectrum
    float* stitch_scratch;

    //! Scratch space, in samples, needed for the stitch_scratch space
    unsigned stitch_scratch_samples;

    float* stitch_scratch_zero_DM;


  };

}

#endif
