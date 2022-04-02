//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/BoreasVoltageUnpackerCUDA.h"
#include "dsp/Operation.h"
#include "dsp/MemoryCUDA.h"

#include "Error.h"

using namespace std;

void check_error_stream (const char*, cudaStream_t);

// unpack FPT input into FPT output
__global__ void boreas_voltage_unpack_fpt (
        float2 * into_p0, float2 * into_p1,
        const float2 * __restrict__ from,
        uint64_t ndat, unsigned chan_stride)
{
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat >= ndat)
    return;

  // offset the input/output indexes to the specified channel
  const uint64_t idx = (blockIdx.y * ndat * 2) + idat;
  const uint64_t odx = (blockIdx.y * chan_stride) + idat;

  // unpack each polarisation
  into_p0[odx] = from[idx];
  into_p1[odx] = from[idx + ndat];
}

CUDA::BoreasVoltageUnpackerEngine::BoreasVoltageUnpackerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

bool CUDA::BoreasVoltageUnpackerEngine::get_device_supported (dsp::Memory* memory) const
{
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
}

void CUDA::BoreasVoltageUnpackerEngine::setup ()
{
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

void CUDA::BoreasVoltageUnpackerEngine::unpack (const dsp::BitSeries * input, dsp::TimeSeries * output)
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const unsigned npol = input->get_npol();

  if (nchan % 2 || npol != 2 || ndim != 2)
    throw Error (InvalidState, "CUDA::BoreasVoltageUnpackerEngine::unpack",
                 "only nchan%2==0, npol==2, ndim==2 supported");

  if (output->get_order() != dsp::TimeSeries::OrderFPT)
    throw Error (InvalidState, "CUDA::BoreasVoltageUnpackerEngine::unpack",
                 "cannot unpack into FPT order");

  // Boreas Voltage Data are packed as Floats in FPT format
  unsigned nthreads = gpu.maxThreadsPerBlock;
  dim3 blocks (ndat / nthreads, nchan, 1);
  if (ndat % nthreads != 0)
    blocks.x++;

  float2 * from = (float2 *) input->get_rawptr();
  float2 * into_p0 = (float2 *) output->get_datptr(0, 0);
  float2 * into_p1 = (float2 *) output->get_datptr(0, 1);
  unsigned chan_stride = ((float2 *) output->get_datptr(1, 0)) - ((float2 *) output->get_datptr(0, 0));

  if (dsp::Operation::verbose)
    cerr << "CUDA::BoreasVoltageUnpackerEngine::unpack from=" << from << " into_p0="
         << into_p0 << " into_p1=" << into_p1 << " ndat=" << ndat << " chan_stride=" << chan_stride << endl;

  boreas_voltage_unpack_fpt<<<blocks, nthreads, 0, stream>>> (into_p0, into_p1, from, ndat, chan_stride);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::BoreasVoltageUnpackerEngine::unpack", stream);
  else
    cudaStreamSynchronize(stream);
}
