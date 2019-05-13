//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_DelayStartTime_h
#define __baseband_cuda_DelayStartTime_h

#include "dsp/DelayStartTime.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class DelayStartTimeEngine : public dsp::DelayStartTime::Engine
  {
  public:

    //! Default Constructor
    DelayStartTimeEngine (cudaStream_t stream);

    ~DelayStartTimeEngine ();

    void set_delays (unsigned npol, unsigned nchan, int64_t delay_samples);

    void delay (const dsp::TimeSeries* in, dsp::TimeSeries* out, uint64_t output_ndat, int64_t delay_samples);

  protected:

    cudaStream_t stream;

    unsigned delay_nchan;

    unsigned delay_npol;

    size_t delays_size;

    int64_t zero_delay;

    int64_t * d_delays;

  };
}

#endif

