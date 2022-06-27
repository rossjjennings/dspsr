/***************************************************************************
 *
 *   Copyright (C) 2022 by Andrew JAmeson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_FloatUnpacker_h
#define __baseband_cuda_FloatUnpacker_h

#include <cuda_runtime.h>

#include "dsp/FloatUnpacker.h"

namespace CUDA
{
  class FloatUnpackerEngine : public dsp::FloatUnpacker::Engine
  {
  public:

    //! Default Constructor
    FloatUnpackerEngine (cudaStream_t stream);

    bool get_device_supported (dsp::Memory* memory) const;

    //! Configure the device
    void setup ();

    //! Unpack the BitSeries to the output timeseries, both in GPU memory
    void unpack (const dsp::BitSeries * input, dsp::TimeSeries * output);

  protected:

    cudaStream_t stream;

    struct cudaDeviceProp gpu;

  };
}

#endif // __baseband_cuda_FloatUnpacker_h