/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/InverseFilterbankConfig.h"
#include "dsp/Scratch.h"
#include "dsp/Dedispersion.h"
#include "dsp/Response.h"

#include "dsp/InverseFilterbankEngineCPU.h"

#if HAVE_CUDA
#include "dsp/InverseFilterbankEngineCUDA.h"
#include "dsp/MemoryCUDA.h"
#endif

#include <iostream>

using namespace std;

dsp::InverseFilterbank::Config::Config ()
{
  memory = 0;
  stream = 0; //(void*)-1;

  nchan = 0; // unspecified. If this stays 0, then no inverse filterbank is applied.
  freq_res = 0;  // unspecified
  when = After;
}

//! Return a new InverseFilterbank instance and configure it
dsp::InverseFilterbank* dsp::InverseFilterbank::Config::create ()
{
  Reference::To<InverseFilterbank> filterbank = new InverseFilterbank;

  filterbank->set_output_nchan( get_nchan() );

  if (freq_res) {
    filterbank->set_input_fft_length(freq_res);
  }

#if HAVE_CUDA
  CUDA::DeviceMemory* device_memory =
    dynamic_cast< CUDA::DeviceMemory*> ( memory );

  if ( device_memory )
  {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>( stream );

    filterbank->set_engine (new CUDA::InverseFilterbankEngineCUDA (cuda_stream));

    Scratch* gpu_scratch = new Scratch;
    gpu_scratch->set_memory (device_memory);
    filterbank->set_scratch (gpu_scratch);
  }
#else
  // default engine is the CPU engine.
  filterbank->set_engine (new dsp::InverseFilterbankEngineCPU);
#endif

  return filterbank.release();
}
