/***************************************************************************
 *
 *   Copyright (C) 2011 by Andrew Jameson & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/TransferBitSeriesCUDA.h"

#include "Error.h"

#include <assert.h>
#include <iostream>
using namespace std;

//! Default constructor- always inplace
dsp::TransferBitSeriesCUDA::TransferBitSeriesCUDA(cudaStream_t _stream)
  : Transformation<BitSeries,BitSeries> ("CUDA::TransferBitSeries", outofplace)
{
  kind = cudaMemcpyHostToDevice;
  stream = _stream;
}

//! Do stuff
void dsp::TransferBitSeriesCUDA::transformation ()
{
  prepare ();

  uint64_t ndat = input->get_ndat();
  uint64_t in_size = input->get_size();
  uint64_t out_size = output->get_size();

  // only transfer data if there is valid data to transfer
  if (ndat > 0)
  {
    if (kind == cudaMemcpyHostToDevice)
    {
      if (stream)
        cudaStreamSynchronize(stream);
      else
        cudaThreadSynchronize();
    }

    if (verbose)
      cerr << "dsp::TransferBitSeriesCUDA::transformation"
           << " ndat=" << ndat
           << " out=" << (void*)output->get_rawptr() << " size=" << out_size
           << " in=" << (void*)input->get_rawptr() << " size=" << in_size
           << endl;

    cudaError error;

    assert (output->get_rawptr() != 0);
    assert (output->get_size() >= input->get_size());

    if (stream)
      error = cudaMemcpyAsync (output->get_rawptr(),
                               input->get_rawptr(),
                               input->get_size(), kind, stream);
    else
      error = cudaMemcpy (output->get_rawptr(),
                          input->get_rawptr(),
                          input->get_size(), kind);

    if (error != cudaSuccess)
      throw Error (InvalidState, "dsp::TransferBitSeriesCUDA::transformation",
                   cudaGetErrorString (error));

    if (kind == cudaMemcpyDeviceToHost)
    {
      if (stream)
        cudaStreamSynchronize(stream);
      else
        cudaThreadSynchronize();
    }
  }
  else
    if (verbose)
      cerr << "dsp::TransferBitSeriesCUDA::transformation skipping transfer as ndat="
           << ndat << endl;
}

void dsp::TransferBitSeriesCUDA::prepare ()
{
  output->internal_match( input );
  output->copy_configuration( input );
}
