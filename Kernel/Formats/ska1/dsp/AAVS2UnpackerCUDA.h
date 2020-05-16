//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Andrew JAmeson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_AAVS2Unpacker_h
#define __baseband_cuda_AAVS2Unpacker_h

#include <cuda_runtime.h>

#include "dsp/AAVS2Unpacker.h"

namespace CUDA
{
  class AAVS2UnpackerEngine : public dsp::AAVS2Unpacker::Engine
  {
  public:

    //! Default Constructor
    AAVS2UnpackerEngine (cudaStream_t stream);

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
