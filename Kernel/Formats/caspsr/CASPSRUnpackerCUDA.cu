//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/CASPSRUnpackerCUDA.h"
#include "dsp/Operation.h"

#include "Error.h"

using namespace std;

void check_error_stream (const char*, cudaStream_t);

__global__ void unpack_real_2pol(uint64_t ndat, float scale, int8_t * from, float* into_pola, float* into_polb)
{
  extern __shared__ int8_t sdata[];

  unsigned idx_shm = threadIdx.x;
  unsigned idx     = (8 * blockIdx.x * blockDim.x) + threadIdx.x;
  unsigned i;

  // each thread will load 8 values (coalesced) from GMEM to SHM
  for (i=0; i<8; i++)
  {
    if (idx < 2*ndat)
    {
      sdata[idx_shm] = from[idx];

      idx     += blockDim.x;
      idx_shm += blockDim.x;
    }
  }

  __syncthreads();

  idx     = (4 * blockIdx.x * blockDim.x) + threadIdx.x;
  idx_shm = threadIdx.x + ((threadIdx.x / 4) * 4);

  // each thread will write 4 values (coalesced) from SHM to GMEM
  for (i=0; i<4; i++)
  {
    if (idx < ndat)
    {
      into_pola[idx] = ((float) sdata[idx_shm]   + 0.5) * scale; 
      into_polb[idx] = ((float) sdata[idx_shm+4] + 0.5) * scale;

      idx += blockDim.x;
      idx_shm += blockDim.x * 2;
    }
  }
}

__global__ void unpack_real_1pol (uint64_t ndat, float scale, int8_t * from, float * into)
{
  const uint64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= ndat)
    return;
  into[idx] = (float(from[idx]) + 0.5) * scale;
}

void caspsr_unpack_2pol (
        cudaStream_t stream, const uint64_t ndat, float scale,
        unsigned char const* input, float* pol0, float* pol1,
        int nthread)
{
  // each thread will unpack 4 time samples from each polarization
  int nsamp_per_block = 4 * nthread;
  int nblock = ndat / nsamp_per_block;
  if (ndat % nsamp_per_block)
    nblock++;

#ifdef _DEBUG
  cerr << "caspsr_unpack ndat=" << ndat << " scale=" << scale 
       << " input=" << (void*) input << " nblock=" << nblock
       << " nthread=" << nthread << endl;
#endif

  int8_t * from = (int8_t *) input;
  size_t shm_bytes = 8 * nthread;
  unpack_real_2pol<<<nblock,nthread,shm_bytes,stream>>> (ndat, scale, from, pol0, pol1);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("caspsr_unpack", stream);
}

void caspsr_unpack_1pol (
        cudaStream_t stream, const uint64_t ndat, float scale,
        unsigned char const* input, float* into,
        int nthread)
{
  dim3 blocks = dim3(ndat / nthread, 1, 1);
  if (ndat % nthread != 0)
    blocks.x++;
#ifdef _DEBUG
  cerr << "caspsr_unpack_1pol ndat=" << ndat << " scale=" << scale
       << " input=" << (void*) input << " nblock=" << nblock
       << " nthread=" << nthread << endl;
#endif

  int8_t * from = (int8_t *) input;
  unpack_real_1pol<<<blocks, nthread, 0, stream>>> (ndat, scale, from, into);

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("unpack_real_1pol", stream);
}
