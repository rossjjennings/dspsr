//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_SampleDelay_h
#define __baseband_cuda_SampleDelay_h

#include "dsp/SampleDelay.h"
#include "dsp/SampleDelayFunction.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class SampleDelayEngine : public dsp::SampleDelay::Engine
  {
  public:

    //! Default Constructor
    SampleDelayEngine (cudaStream_t stream);

    ~SampleDelayEngine ();

    void set_delays (unsigned npol, unsigned nchan, std::vector<int64_t> zero_delay,
                     dsp::SampleDelayFunction * function);

    void retard (const dsp::TimeSeries* in, dsp::TimeSeries* out, uint64_t output_ndat);

  protected:

    cudaStream_t stream;

    unsigned delay_nchan;

    unsigned delay_npol;

    size_t delays_size;

    int64_t * d_delays;

  };
}

#endif
