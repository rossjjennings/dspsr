//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2009 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/General/dsp/InverseFilterbankEngineCPU.h

#ifndef __InverseFilterbankCPU_h
#define __InverseFilterbankCPU_h

#include "dsp/InverseFilterbankEngine.h"
#include "dsp/LaunchConfig.h"

namespace dsp
{
  class elapsed
  {
  public:
    elapsed ();
    void wrt (cudaEvent_t before);

    double total;
    cudaEvent_t after;
  };

  class InverseFilterbankEngineCPU : public dsp::InverseFilterbank::Engine
  {
    unsigned nstream;

  public:

    //! Default Constructor
    InverseFilterbankEngine (cudaStream_t stream);

    ~InverseFilterbankEngine ();

    void setup (dsp::Filterbank*);
    void set_scratch (float *);

    void perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
                  uint64_t npart, uint64_t in_step, uint64_t out_step);

    void finish ();

  protected:

    //! forward fft plan
    cufftHandle plan_fwd;

    //! backward fft plan
    cufftHandle plan_bwd;

    //! Complex-valued data
    bool real_to_complex;

    //! inplace FFT in CUDA memory
    float2* d_fft;

    //! convolution kernel in CUDA memory
    float2* d_kernel;

    //! device scratch sapce
    float* scratch;

    unsigned nchan_subband;
    unsigned freq_res;
    unsigned nfilt_pos;
    unsigned nkeep;

    LaunchConfig1D multiply;

    cudaStream_t stream;

    bool verbose;

  };

}

#endif
