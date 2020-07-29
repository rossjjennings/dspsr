//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2020 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/LFAASPEADUnpackerCUDA.h"
#include "dsp/Operation.h"

//#include "dsp/MemoryCUDA.h"

#include "Error.h"

using namespace std;

void check_error_stream (const char*, cudaStream_t);

__global__ void lfaaspead_unpack_fpt (uint64_t ndat, float scale,
                                      float2 * into_p0, float2 * into_p1,
                                      const int32_t * from, unsigned out_chan_stride)
{
  // heap length = 2048
  // ichan == blockIdx.y
  // iheap == blockIdx.x
  const unsigned heap_size = 2048;
  const unsigned ndimpol = 2 * 2;
  const unsigned ichan = blockIdx.y;
  const unsigned heap_idat = blockIdx.x * heap_size;

  //       iheap      * nchan     * 2048       +  ichan      * 2048
  from += (blockIdx.x * gridDim.y * heap_size) + (blockIdx.y * heap_size);
  //          ichan      * chan_stride
  into_p0 += (blockIdx.y * out_chan_stride) + (blockIdx.x * heap_size);
  into_p1 += (blockIdx.y * out_chan_stride) + (blockIdx.x * heap_size);

  // isamp is the value within the heap
  for (unsigned isamp=threadIdx.x; isamp<heap_size; isamp+=blockDim.x)
  {
    if (heap_idat + isamp >= ndat)
      return;

    // read Re(P0), Im(P0), Re(P1), Im(P1) for 1 sample
    int32_t in32 = from[isamp];
    int8_t * in8 = (int8_t *) &in32;

    into_p0[isamp].x = scale * float(in8[0]);
    into_p0[isamp].y = scale * float(in8[1]);

    into_p1[isamp].x = scale * float(in8[2]);
    into_p1[isamp].y = scale * float(in8[3]);
  }
}

CUDA::LFAASPEADUnpackerEngine::LFAASPEADUnpackerEngine (cudaStream_t _stream)
{
  cerr << "CUDA::LFAASPEADUnpackerEngine::LFAASPEADUnpackerEngine ctor" << endl;
  stream = _stream;
}

void CUDA::LFAASPEADUnpackerEngine::setup ()
{
  cerr << "CUDA::LFAASPEADUnpackerEngine::setup()" << endl;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

void CUDA::LFAASPEADUnpackerEngine::unpack (float scale, const dsp::BitSeries * input, dsp::TimeSeries * output)
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const unsigned npol = input->get_npol();

  if (npol != 2 || ndim != 2)
    throw Error (InvalidState, "CUDA::LFAASPEADUnpackerEngine::unpack",
                 "only npol==2, ndim==2 supported");

  if (output->get_order() != dsp::TimeSeries::OrderFPT)
    throw Error (InvalidState, "CUDA::LFAASPEADUnpackerEngine::unpack",
                 "cannot unpack into FPT order");

  // input is packed in heaps of FPT ordered data
  unsigned nsamp_per_heap = 2048;
  unsigned nheap = ndat / nsamp_per_heap;
  if (ndat % nsamp_per_heap != 0)
    throw Error (InvalidState, "CUDA::LFAASPEADUnpackerEngine::unpack",
                 "ndat=%ul was not a multiple of nsamp_per_heap=%u", ndat, nsamp_per_heap);

  dim3 blocks = dim3(nheap, nchan, 1);
  unsigned nthreads = gpu.maxThreadsPerBlock;

  void * from = (void *) input->get_rawptr();
  float * into_p0 = (float *) output->get_datptr(0, 0);
  float * into_p1 = (float *) output->get_datptr(0, 1);
  if (dsp::Operation::verbose)
    cerr << "CUDA::LFAASPEADUnpackerEngine::unpack from=" << from << " into_p0="
         << into_p0 << " into_p1=" << into_p1 << endl;

  // difference between chan 0 and 1 in 
  unsigned out_chan_stride = 0;
  if (nchan > 1)
    out_chan_stride = ((float *) output->get_datptr(1, 0) - (float *) output->get_datptr(0, 0)) / ndim;

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::LFAASPEADUnpackerEngine::unpack nthreads=" << nthreads << endl;
    cerr << "CUDA::LFAASPEADUnpackerEngine::unpack blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << ")" << endl;
    cerr << "CUDA::LFAASPEADUnpackerEngine::unpack ndat=" << ndat << " scale=" << scale << " out_chan_stride=" << out_chan_stride << endl;
  }

  lfaaspead_unpack_fpt<<<blocks, nthreads, 0, stream>>> (ndat, scale, (float2 *) into_p0, (float2 *) into_p1, (const int32_t *) from, out_chan_stride);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::LFAASPEADUnpackerEngine::unpack", stream);
  else
    cudaStreamSynchronize(stream);
}
