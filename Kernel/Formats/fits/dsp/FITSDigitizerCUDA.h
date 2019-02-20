//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#ifndef __FITSDigitizerEngine_h
#define __FITSDigitizerEngine_h

#include "dsp/FITSDigitizer.h"
#include "dsp/Scratch.h"

#include <cuda_runtime.h>

namespace CUDA
{
  class FITSDigitizerEngine : public dsp::FITSDigitizer::Engine
  {
  public:

    FITSDigitizerEngine (cudaStream_t stream);

    ~FITSDigitizerEngine ();

    void set_scratch (dsp::Scratch * scratch);

    void set_rescale_nblock (const dsp::TimeSeries * input, unsigned rescale_nblock);

    void set_mapping (const dsp::TimeSeries * input,
                      dsp::ChannelSort& channel);

    void measure_scale (const dsp::TimeSeries * input,
                        unsigned rescale_nsamp);

    void digitize (const dsp::TimeSeries * input, dsp::BitSeries * output, 
                   uint64_t ndat, unsigned nbit, float digi_mean, 
                   float digi_scale, int digi_min, int digi_max);

    void get_scale_offsets (double * scale, double * offset,
                            unsigned nchan, unsigned npol);

  protected:

    cudaStream_t stream;

  private:

    dsp::Scratch * scratch;

    unsigned rescale_nblock;

    float * freq_total;

    float * freq_totalsq;

    size_t freq_total_size;

    float * d_scale;

    float * d_offset;

    float * h_scale;

    float * h_offset;

    size_t scale_offset_size;

    unsigned * mapping;

    size_t mapping_size;
  };
}

#endif // __FITSDigitizerEngine_h
