//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/AAVS2UnpackerCUDA.h"
#include "dsp/Operation.h"

//#include "dsp/MemoryCUDA.h"

#include "Error.h"

using namespace std;

void check_error_stream (const char*, cudaStream_t);

// unpack TFP input into FPT output
__global__ void aavs2_unpack_fpt (uint64_t ndat, float scale,
                                  float2 * into_p0, float2 * into_p1,
                                  const int32_t * from)
{
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat >= ndat)
    return;

  // read Re(P0), Im(P0), Re(P1), Im(P1) for 1 sample
  int32_t in32 = from[idat];
  int8_t * in8 = (int8_t *) &in32;

  for (unsigned i=0; i<4; i++)
  {
    if ((in8[i] == -128) || (in8[i] == -127))
      in8[i] = 0;
  }

  into_p0[idat].x = scale * float(in8[0]);
  into_p0[idat].y = scale * float(in8[1]);

  into_p1[idat].x = scale * float(in8[2]);
  into_p1[idat].y = scale * float(in8[3]);
}

CUDA::AAVS2UnpackerEngine::AAVS2UnpackerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::AAVS2UnpackerEngine::setup ()
{
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

void CUDA::AAVS2UnpackerEngine::unpack (float scale, const dsp::BitSeries * input, dsp::TimeSeries * output)
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const unsigned npol = input->get_npol();

  if (nchan != 1 || npol != 2 || ndim != 2)
    throw Error (InvalidState, "CUDA::AAVS2UnpackerEngine::unpack",
                 "only nchan==1, npol==2, ndim==2 supported");

  if (output->get_order() != dsp::TimeSeries::OrderFPT)
    throw Error (InvalidState, "CUDA::AAVS2UnpackerEngine::unpack",
                 "cannot unpack into FPT order");

  unsigned nthreads = gpu.maxThreadsPerBlock;
  unsigned nblocks = ndat / nthreads;
  if (ndat % nthreads)
    nblocks++;

  void * from = (void *) input->get_rawptr();
  float * into_p0 = (float *) output->get_datptr(0, 0);
  float * into_p1 = (float *) output->get_datptr(0, 1);
  if (dsp::Operation::verbose)
    cerr << "CUDA::AAVS2UnpackerEngine::unpack from=" << from << " into_p0="
         << into_p0 << " into_p1=" << into_p1 << endl;

  aavs2_unpack_fpt<<<nblocks, nthreads, 0, stream>>> (ndat, scale, (float2 *) into_p0, (float2 *) into_p1, (int32_t *) from);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::AAVS2UnpackerEngine::unpack", stream);
  else
    cudaStreamSynchronize(stream);
}
