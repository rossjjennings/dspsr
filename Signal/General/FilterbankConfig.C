/***************************************************************************
 *
 *   Copyright (C) 2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/FilterbankConfig.h"
#include "dsp/Scratch.h"

#include "dsp/FilterbankEngineCPU.h"

#if HAVE_CUDA
#include "dsp/FilterbankEngineCUDA.h"
#include "dsp/MemoryCUDA.h"
#endif

#include <iostream>
using namespace std;


dsp::Filterbank::Config::Config ()
{
  memory = 0;
  stream = 0; //(void*)-1;

  nchan = 1;
  freq_res = 0;  // unspecified
  when = After;  // not good, but the original default
}

//! Return a new Filterbank instance and configure it
dsp::Filterbank* dsp::Filterbank::Config::create ()
{
  Reference::To<Filterbank> filterbank = new Filterbank;

  filterbank->set_nchan( get_nchan() );

  if (freq_res)
    filterbank->set_frequency_resolution ( freq_res );



#if HAVE_CUDA
  CUDA::DeviceMemory* device_memory =
    dynamic_cast< CUDA::DeviceMemory*> ( memory );

  if ( device_memory )
  {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>( stream );

    filterbank->set_engine (new CUDA::FilterbankEngineCUDA (cuda_stream));

    Scratch* gpu_scratch = new Scratch;
    gpu_scratch->set_memory (device_memory);
    filterbank->set_scratch (gpu_scratch);
  }
#else
  filterbank->set_engine (new FilterbankEngineCPU);
#endif

  return filterbank.release();
}
