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

  nchan = 1;
  freq_res = 0;  // unspecified
  when = During;  // not good, but the original default
}

std::ostream& dsp::operator << (std::ostream& os,
				const dsp::InverseFilterbank::Config& config)
{
  os << config.get_nchan();
  if (config.get_convolve_when() == InverseFilterbank::Config::Before)
    os << ":B";
  else if (config.get_convolve_when() == InverseFilterbank::Config::During)
    os << ":D";
  else if (config.get_freq_res() != 1)
    os << ":" << config.get_freq_res();

  return os;
}

//! Extraction operator
std::istream& dsp::operator >> (std::istream& is, dsp::InverseFilterbank::Config& config)
{
  unsigned value;
  is >> value;

  config.set_nchan (value);
  config.set_convolve_when (InverseFilterbank::Config::After);

  if (is.eof())
    return is;

  if (is.peek() != ':')
  {
    is.fail();
    return is;
  }

  // throw away the colon
  is.get();

  if (is.peek() == 'D' || is.peek() == 'd')
  {
    is.get();  // throw away the D
    config.set_convolve_when (InverseFilterbank::Config::During);
  }
  else if (is.peek() == 'B' || is.peek() == 'b')
  {
    is.get();  // throw away the B
    config.set_convolve_when (InverseFilterbank::Config::Before);
  }
  else
  {
    unsigned nfft;
    is >> nfft;
    config.set_freq_res (nfft);
  }

  return is;
}

//! Set the device on which the unpacker will operate
void dsp::InverseFilterbank::Config::set_device (Memory* mem)
{
  memory = mem;
}

void dsp::InverseFilterbank::Config::set_stream (void* ptr)
{
  stream = ptr;
}

//! Return a new InverseFilterbank instance and configure it
dsp::InverseFilterbank* dsp::InverseFilterbank::Config::create ()
{
  Reference::To<InverseFilterbank> filterbank = new InverseFilterbank;

  filterbank->set_output_nchan( get_nchan() );

  if (freq_res){
    filterbank->set_frequency_resolution ( freq_res );
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
// #else
//   filterbank->set_engine (new dsp::InverseFilterbankEngineCPU);
#endif

  return filterbank.release();
}
