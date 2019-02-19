//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/FilterbankEngineCPU.h

#ifndef __FilterbankEngineCPU_h
#define __FilterbankEngineCPU_h

#include "dsp/FilterbankEngine.h"

namespace dsp
{
  //! Discrete convolution filterbank step for CPU
  class FilterbankEngineCPU : public Filterbank::Engine
  {
  public:

    //! Default Constructor
    FilterbankEngineCPU ();

    ~FilterbankEngineCPU ();

    void setup (dsp::Filterbank*);

    void set_scratch (float *);

    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);

    void finish ();

    // virtual void set_passband (dsp::Response* _passband);

    // virtual const dsp::Response* get_passband () const;


  protected:

    FTransform::Plan* forward;

    FTransform::Plan* backward;

    //! Complex-valued data
    bool real_to_complex;

    //! device scratch sapce
    float* scratch;

    //! scratch space for forward fft
    //! This is an array of float pointers because
    //! we might be dealing not only with multiple
    //! polarizations, but also with cross polarization.
    float* freq_domain_scratch[2];

    //! scratch space for backward fft
    float* time_domain_scratch;

    //! scratch space for apodization operation
    float* windowed_time_domain_scratch;


    //! response kernel, from Filterbank
    const dsp::Response* response;

    //! apodization kernel, from Filterbank
    const dsp::Apodization* apodization;

    //! Whether or not to do matrix convolution, from Filterbank
    bool matrix_convolution;

    //! number of output channels per input channel
    unsigned nchan_subband;

    //! frequency resolution of response (dsp::Response::get_ndat)
    unsigned freq_res;

    //! positive impulse from response (dsp::Response::get_impulse_pos)
    unsigned nfilt_pos;

    //! negative impulse from response (dsp::Response::get_impulse_neg)
    unsigned nfilt_neg;

    //! number of samples in forward fft
    uint64_t nsamp_fft;


    //! number of samples to keep from each input sample.
    //! This is essentially the number of fft points minus the total
    //! smearing from the response
    unsigned nkeep;

    bool verbose;

  };

}

#endif
