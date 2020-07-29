//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Andrew JAmeson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_LFAASPEADUnpacker_h
#define __baseband_cuda_LFAASPEADUnpacker_h

#include <cuda_runtime.h>

#include "dsp/LFAASPEADUnpacker.h"

namespace CUDA
{
  class LFAASPEADUnpackerEngine : public dsp::LFAASPEADUnpacker::Engine
  {
  public:

    //! Default Constructor
    LFAASPEADUnpackerEngine (cudaStream_t stream);

    //! Configure the device
    void setup ();

    //! Unpack the BitSeries to the output timeseries, both in GPU memory
    void unpack (float scale, const dsp::BitSeries * input, dsp::TimeSeries * output);

  protected:

    cudaStream_t stream;

    struct cudaDeviceProp gpu;
  };
}
#endif
