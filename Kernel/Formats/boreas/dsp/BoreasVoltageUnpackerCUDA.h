/***************************************************************************
 *
 *   Copyright (C) 2022 by Andrew JAmeson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __baseband_cuda_BoreasVoltageUnpacker_h
#define __baseband_cuda_BoreasVoltageUnpacker_h

#include <cuda_runtime.h>

#include "dsp/BoreasVoltageUnpacker.h"

namespace CUDA
{
  class BoreasVoltageUnpackerEngine : public dsp::BoreasVoltageUnpacker::Engine
  {
  public:

    //! Default Constructor
    BoreasVoltageUnpackerEngine (cudaStream_t stream);

    //! Return true if specified memory is CUDA Device memory 
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

#endif
