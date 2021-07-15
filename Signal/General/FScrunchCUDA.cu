//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FScrunchCUDA.h"

#include "Error.h"
#include "debug.h"

#include <cuComplex.h>

using namespace std;

void check_error (const char*);

CUDA::FScrunchEngine::FScrunchEngine (cudaStream_t _stream)
{
  stream = _stream;
}

__inline__ __device__ float sumTwo(float v1, float v2)
{
  return v1 + v2;
}

__inline__ __device__ float2 sumTwo(float2 v1, float2 v2)
{
  return cuCaddf(v1, v2);
}

__inline__ __device__ float4 sumTwo(float4 v1, float4 v2)
{
  float4 r;
  r.w = v1.w + v2.w;
  r.x = v1.x + v2.x;
  r.y = v1.y + v2.y;
  r.z = v1.z + v2.z;
  return r;
}

template<typename T>
__global__ void fpt_fscrunch_kernel (T* in_base, T* out_base,
    unsigned in_Fstride, unsigned in_Pstride,
    unsigned out_Fstride, unsigned out_Pstride,
    unsigned ndat, unsigned sfactor)
{

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ndat)
    return;

  uint64_t idx = (blockIdx.y * in_Fstride * sfactor) + (blockIdx.z * in_Pstride) + i;

  T result = in_base[idx];
  for (int j=1; j < sfactor; ++j)
  {
    idx += in_Fstride;
    result = sumTwo(result, in_base[idx]);
  }

  out_base[(blockIdx.y * out_Fstride) + (blockIdx.z * out_Pstride) + i] = result;
}

void CUDA::FScrunchEngine::fpt_fscrunch(const dsp::TimeSeries *in,
    dsp::TimeSeries* out, unsigned sfactor)
{
  uint64_t ndat = in->get_ndat ();
  unsigned ndim = in->get_ndim();
  unsigned npol = in->get_npol();
  unsigned nchan = in->get_nchan();
  unsigned nchan_out = nchan / sfactor;

  if (dsp::Operation::verbose)
    cerr << "CUDA::FScrunchEngine::fpt_fscrunch ndat=" << ndat << " ndim=" 
         << ndim << " npol=" << npol << " nchan=" << nchan << " nchan_out="
         << nchan_out << endl;

  if (ndat == 0)
  {
    return;
  }

  // set up two-dimensional blocks; the first dimension corresponds to an
  // index along the data rows (so polarization, time, and dimension), and
  // the second to output channel; each thread will add up the rows for
  // sfactor input channels and write out to a single output channel; this
  // avoid any need for synchronization

  // TODO -- enforce out of place?  Technically could work with in place.

  uint64_t in_Fstride = 0;
  uint64_t out_Fstride = 0;
  if (nchan > 1)
  {
    in_Fstride = (in->get_datptr(1,0)-in->get_datptr(0,0)) / ndim;
    out_Fstride = (out->get_datptr(1,0)-out->get_datptr(0,0)) / ndim;
  }

  uint64_t in_Pstride = 0;
  uint64_t out_Pstride = 0;
  if (npol > 1)
  {
    in_Pstride = (in->get_datptr(0,1)-in->get_datptr(0,0)) / ndim;
    out_Pstride = (out->get_datptr(0,1)-out->get_datptr(0,0)) / ndim;
  }

  dim3 threads (128);
  dim3 blocks (ndat/threads.x, nchan_out, npol);

  if (ndat % threads.x)
    blocks.x ++;

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::FScrunchEngine::fpt_fscrunch blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << ")" << endl;
    cerr << "CUDA::FScrunchEngine::fpt_fscrunch threads=(" << threads.x << "," << threads.y << "," << threads.z << ")" << endl;
    cerr << "CUDA::FScrunchEngine::fpt_fscrunch Fstride in=" << in_Fstride << " out=" << out_Fstride << endl;
    cerr << "CUDA::FScrunchEngine::fpt_fscrunch Pstride in=" << in_Pstride << " out=" << out_Pstride << endl;
  }

  void * in_base = (void *) in->get_datptr(0,0);
  void * out_base = (void *) out->get_datptr(0,0);

  if (ndim == 1)
  {
    fpt_fscrunch_kernel<float><<<blocks,threads,0,stream>>> (
      (float*)(in_base), (float*)(out_base), 
      in_Fstride, in_Pstride, out_Fstride, out_Pstride,
      ndat, sfactor);
  }
  else if (ndim == 2)
  {
    fpt_fscrunch_kernel<float2><<<blocks,threads,0,stream>>> (
      (float2*)(in_base), (float2*)(out_base), 
      in_Fstride, in_Pstride, out_Fstride, out_Pstride,
      ndat, sfactor);
  }
  else if (ndim == 4)
  {
    fpt_fscrunch_kernel<float4><<<blocks,threads,0,stream>>> (
      (float4*)(in_base), (float4*)(out_base), 
      in_Fstride, in_Pstride, out_Fstride, out_Pstride,
      ndat, sfactor);
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::FScrunchEngine::fpt_scrunch");
}

