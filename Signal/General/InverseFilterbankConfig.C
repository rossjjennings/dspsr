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

namespace dsp {

  InverseFilterbank::Config::Config ()
  {
    memory = 0;
    stream = 0; //(void*)-1;

    nchan = 0; // unspecified. If this stays 0, then no inverse filterbank is applied.
    freq_res = 0;  // unspecified
    input_overlap = 0;
    when = After;
  }

  //! Return a new InverseFilterbank instance and configure it
  InverseFilterbank* InverseFilterbank::Config::create ()
  {
    Reference::To<InverseFilterbank> filterbank = new InverseFilterbank;

    filterbank->set_output_nchan( get_nchan() );

    if (freq_res) {
      filterbank->set_input_fft_length(freq_res);
    }

    // if (input_overlap) {
    //   filterbank->set_input_overlap(input_overlap);
    // }
    filterbank->set_engine (new InverseFilterbankEngineCPU);

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
  #endif
    return filterbank.release();
  }

  //! Insertion operator
  std::ostream& operator << (std::ostream& os, const InverseFilterbank::Config& config)
  {
    os << config.get_nchan();
    Filterbank::Config::When convolve_when = config.get_convolve_when();

    if (convolve_when == Filterbank::Config::Before) {
      os << ":B";
    } else if (convolve_when == Filterbank::Config::During) {
      os << ":D";
    } else if (config.get_freq_res() != 1){
      os << ":" << config.get_freq_res();
    }

    unsigned overlap = config.get_input_overlap();
    if (overlap != 0) {
      os << ":" << overlap;
    }
    return os;
  }

  //! Extraction operator
  std::istream& operator >> (std::istream& is, InverseFilterbank::Config& config)
  {
    unsigned value;
    is >> value;

    config.set_nchan (value);
    config.set_convolve_when (Filterbank::Config::After);

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
      config.set_convolve_when (Filterbank::Config::During);
    }
    else if (is.peek() == 'B' || is.peek() == 'b')
    {
      is.get();  // throw away the B
      config.set_convolve_when (Filterbank::Config::Before);
    }
    else if (is.peek() == 'A' || is.peek() == 'a')
    {
      // even though this is the default, make it possible to pass A anyways.
      is.get();  // throw away the A
      config.set_convolve_when (Filterbank::Config::After);
    }
    else
    {
      unsigned nfft;
      is >> nfft;
      config.set_freq_res (nfft);
    }

    if (is.peek() == ':') {
      is.get();
      unsigned overlap;
      is >> overlap;
      config.set_input_overlap (overlap);
    }

    return is;
  }
}
