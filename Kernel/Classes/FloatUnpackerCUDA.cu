//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2022 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FloatUnpackerCUDA.h"
#include "dsp/Operation.h"
#include "dsp/MemoryCUDA.h"
 
#include "Error.h"
 
using namespace std;
 
void check_error_stream (const char*, cudaStream_t);
 
// unpack single pol TFP input into FPT output
template<typename T>
__global__ void float_unpack_1pol_fpt (
    T * into, const T* __restrict__ from,
    uint64_t ndat, unsigned nchan, unsigned ochan_stride)
{
  // shared storage for in-block cornerturn
  __shared__ T float_unpack_1pol_fpt[32][33];

  // input is TF ordered output is FT ordered
  const int ichan = blockIdx.x * blockDim.x + threadIdx.x;
  const int idat = blockIdx.y * blockDim.y + threadIdx.y;

  // read 32 samples and 32 channels of input into shared memory in TF order
  if (idat < ndat && ichan < nchan)
    float_unpack_1pol_fpt[threadIdx.y][threadIdx.x] = from[(idat * nchan) + ichan];

  __syncthreads();

  // write 32 channels and 32 samples in FT ordering
  const int odat = blockIdx.y * blockDim.y + threadIdx.x;
  const int ochan = blockIdx.x * blockDim.x + threadIdx.y;
  if (odat >= ndat || ochan >= nchan)
    return;
  into[(ochan * ochan_stride) + odat] = float_unpack_1pol_fpt[threadIdx.x][threadIdx.y];
}

// unpack dual pol TFP input into FPT output
__global__ void float_unpack_2pol_1dim_fpt (
  float * into_p0, float * into_p1, const float2 * __restrict__ from,
  uint64_t ndat, unsigned nchan, unsigned ochan_stride)
{
  // shared storage for in-block cornerturn
  __shared__ float2 float_unpack_2pol_1dim_fpt[32][33];

  const int ichan = blockIdx.x * blockDim.x + threadIdx.x;
  const int idat = blockIdx.y * blockDim.y + threadIdx.y;
  if (idat < ndat && ichan < nchan)
    float_unpack_2pol_1dim_fpt[threadIdx.y][threadIdx.x] = from[(idat * nchan) + ichan];
  __syncthreads();

  const int odat = blockIdx.y * blockDim.y + threadIdx.x;
  const int ochan = blockIdx.x * blockDim.x + threadIdx.y;
  if (odat >= ndat || ochan >= nchan)
    return;

  const int odx = ochan * ochan_stride + odat;
  into_p0[odx] = float_unpack_2pol_1dim_fpt[threadIdx.x][threadIdx.y].x;
  into_p1[odx] = float_unpack_2pol_1dim_fpt[threadIdx.x][threadIdx.y].y;
}

// unpack dual pol TFP input into FPT output
__global__ void float_unpack_2pol_2dim_fpt (
    float2 * into_p0, float2 * into_p1, const float4 * __restrict__ from,
    uint64_t ndat, unsigned nchan, unsigned ochan_stride)
{
  // shared storage for in-block cornerturn
  __shared__ float4 float_unpack_2pol_2dim_fpt[32][33];

  const int ichan = blockIdx.x * blockDim.x + threadIdx.x;
  const int idat = blockIdx.y * blockDim.y + threadIdx.y;
  if (idat < ndat && ichan < nchan)
    float_unpack_2pol_2dim_fpt[threadIdx.y][threadIdx.x] = from[(idat * nchan) + ichan];
  __syncthreads();

  const int odat = blockIdx.y * blockDim.y + threadIdx.x;
  const int ochan = blockIdx.x * blockDim.x + threadIdx.y;
  if (odat >= ndat || ochan >= nchan)
    return;

  const int odx = ochan * ochan_stride + odat;
  into_p0[odx] = make_float2(float_unpack_2pol_2dim_fpt[threadIdx.x][threadIdx.y].x, float_unpack_2pol_2dim_fpt[threadIdx.x][threadIdx.y].y);
  into_p1[odx] = make_float2(float_unpack_2pol_2dim_fpt[threadIdx.x][threadIdx.y].z, float_unpack_2pol_2dim_fpt[threadIdx.x][threadIdx.y].w);
}

// unpack N pol TFP input into FPT output
__global__ void float_unpack_4pol_1dim_fpt (
    float * into, const float4 * __restrict__ from,
    uint64_t ndat, unsigned nchan, unsigned ochan_stride, unsigned opol_stride)
{
  // shared storage (TFP order) for in-block cornerturn
  __shared__ float4 float_unpack_4pol_1dim_fpt[32][33];

  // reads 32 samples, 32 channels, npol pols into shared memory
  const int ichan = blockIdx.x * blockDim.x + threadIdx.x;
  const int idat = blockIdx.y * blockDim.y + threadIdx.y;
  if (idat < ndat && ichan < nchan)
    float_unpack_4pol_1dim_fpt[threadIdx.y][threadIdx.x] = from[(idat * nchan) + ichan];

  __syncthreads();

  // thread X now indexes on dat, thread y on chan
  const int odat = blockIdx.y * blockDim.y + threadIdx.x;
  const int ochan = blockIdx.x * blockDim.x + threadIdx.y;
  if (odat >= ndat || ochan >= nchan)
    return;

  int odx = (ochan * ochan_stride) + odat;
  into[odx] = float_unpack_4pol_1dim_fpt[threadIdx.x][threadIdx.y].x;
  odx += opol_stride;
  into[odx] = float_unpack_4pol_1dim_fpt[threadIdx.x][threadIdx.y].y;
  odx += opol_stride;
  into[odx] = float_unpack_4pol_1dim_fpt[threadIdx.x][threadIdx.y].z;
  odx += opol_stride;
  into[odx] = float_unpack_4pol_1dim_fpt[threadIdx.x][threadIdx.y].w;
}

CUDA::FloatUnpackerEngine::FloatUnpackerEngine (cudaStream_t _stream)
{
  stream = _stream;
}

bool CUDA::FloatUnpackerEngine::get_device_supported (dsp::Memory* memory) const
{
  return dynamic_cast< CUDA::DeviceMemory*> ( memory );
}

void CUDA::FloatUnpackerEngine::setup ()
{
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties (&gpu, device);
}

void CUDA::FloatUnpackerEngine::unpack (const dsp::BitSeries * input, dsp::TimeSeries * output)
{
  const uint64_t ndat = input->get_ndat();
  const unsigned nchan = input->get_nchan();
  const unsigned npol = input->get_npol();
  const unsigned ndim = input->get_ndim();
 
  if (dsp::Operation::verbose)
    cerr << "CUDA::FloatUnpackerEngine::unpack ndat=" << ndat << " nchan=" << nchan << " npol=" << npol << " ndim=" << ndim << endl;

  if (output->get_order() == dsp::TimeSeries::OrderTFP)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::FloatUnpackerEngine::unpack output->get_order()=dsp::TimeSeries::OrderTFP" << endl;
    uint64_t nbytes = ndat * nchan * npol * ndim * sizeof(float);
    void * from = (void *) input->get_rawptr();
    void * to = (void *) output->get_dattfp();
    cudaError_t result = cudaMemcpyAsync(to, from, nbytes, cudaMemcpyDeviceToDevice, stream);
    if (result != cudaSuccess)
    {
      cudaStreamSynchronize(stream);      
      throw Error (InvalidState, "CUDA::FloatUnpackerEngine::unpack", cudaGetErrorString(result));
    }
  }
  else if (output->get_order() == dsp::TimeSeries::OrderFPT)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::FloatUnpackerEngine::unpack output->get_order()=dsp::TimeSeries::OrderFPT" << endl;
    dim3 threads(32, 32, 1);
    dim3 blocks (nchan / threads.x, ndat / threads.y, 1);
    if (nchan % threads.x != 0)
      blocks.x++;
    if (ndat % threads.y != 0)
      blocks.y++;
    unsigned ochan_stride = 0;

    if (dsp::Operation::verbose)
      cerr << "CUDA::FloatUnpackerEngine::unpack blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << ") "
           << "threads=(" << threads.x << "," << threads.y << "," << threads.z << ")" << endl;
    if (npol == 4)
    {
      if (ndim == 1)
      {
        float4 * from = (float4 *) input->get_rawptr();
        float * into = output->get_datptr(0, 0);
        unsigned opol_stride = output->get_datptr(0, 1) - into;
        if (nchan > 1)
          ochan_stride =  output->get_datptr(1, 0) - into; 

        if (dsp::Operation::verbose)
          cerr << "CUDA::FloatUnpackerEngine::unpack ochan_stride=" << ochan_stride << " opol_stride=" << opol_stride << endl;
        float_unpack_4pol_1dim_fpt<<<blocks, threads, 0, stream>>>(into, from, ndat, nchan, ochan_stride, opol_stride);
        if (dsp::Operation::record_time || dsp::Operation::verbose)
          check_error_stream ("CUDA::FloatUnpackerEngine::unpack::float_unpack_4pol_1dim_fpt", stream);
      }
      else
        throw Error (InvalidState, "CUDA::FloatUnpackerEngine::unpack", "unsupported dims: npol=4, ndim=%u", ndim);
    }
    else if (npol == 2)
    {
      if (ndim == 1)
      {
        float2 * from = (float2 *) input->get_rawptr();
        float * into_p0 = output->get_datptr(0, 0);
        float * into_p1 = output->get_datptr(0, 1);
        if (nchan > 1)
          ochan_stride = output->get_datptr(1, 0) - into_p0;

        float_unpack_2pol_1dim_fpt<<<blocks, threads, 0, stream>>>(into_p0, into_p1, from, ndat, nchan, ochan_stride);

        if (dsp::Operation::record_time || dsp::Operation::verbose)
          check_error_stream ("CUDA::FloatUnpackerEngine::unpack::float_unpack_2pol_1dim_fpt", stream);
      }
      else if (ndim == 2)
      {
        float4 * from = (float4 *) input->get_rawptr();
        float2 * into_p0 = (float2 *) output->get_datptr(0, 0);
        float2 * into_p1 = (float2 *) output->get_datptr(0, 1);
        if (nchan > 1)
          ochan_stride = ((float2 *) output->get_datptr(1, 0)) - into_p0;

        float_unpack_2pol_2dim_fpt<<<blocks, threads, 0, stream>>>(into_p0, into_p1, from, ndat, nchan, ochan_stride);
        if (dsp::Operation::record_time || dsp::Operation::verbose)
          check_error_stream ("CUDA::FloatUnpackerEngine::unpack::float_unpack_2pol_2dim_fpt", stream);
      }
      else
        throw Error (InvalidState, "CUDA::FloatUnpackerEngine::unpack", "unsupported dims: npol=2, ndim=%u", ndim);
    }
    else if (npol == 1)
    {
      unsigned nthreads = gpu.maxThreadsPerBlock;
      dim3 blocks (ndat / nthreads, nchan, 1);
      if (ndat % nthreads != 0)
        blocks.x++;
    
      if (ndim == 1)
      {
        float * from = (float *) input->get_rawptr();
        float * into = output->get_datptr(0, 0);
        if (nchan > 1)
          ochan_stride = (output->get_datptr(1, 0) - into);

        float_unpack_1pol_fpt<<<blocks, threads, 0, stream>>>(into, from, ndat, nchan, ochan_stride);
        if (dsp::Operation::record_time || dsp::Operation::verbose)
          check_error_stream ("CUDA::FloatUnpackerEngine::unpack::float_unpack_1pol_fpt<float>", stream);
      }
      else if (ndim == 2)
      {
        float2 * from = (float2 *) input->get_rawptr();
        float2 * into = (float2 *) output->get_datptr(0, 0);
        if (nchan > 1)
          ochan_stride = ((float2 *) output->get_datptr(1, 0)) - into;
        float_unpack_1pol_fpt<<<blocks, threads, 0, stream>>>(into, from, ndat, nchan, ochan_stride); 
        if (dsp::Operation::record_time || dsp::Operation::verbose)
          check_error_stream ("CUDA::FloatUnpackerEngine::unpack::float_unpack_1pol_fpt<float2>", stream);
      }
      else
        throw Error (InvalidState, "CUDA::FloatUnpackerEngine::unpack", "unsupported dims: npol=1, ndim=%u", ndim);
    }
    else
      throw Error (InvalidState, "CUDA::FloatUnpackerEngine::unpack", "unsupported dims: npol=%u", npol);

  }
  // stream synchronization is required for unpackers
  if (!(dsp::Operation::record_time || dsp::Operation::verbose))
    cudaStreamSynchronize(stream);      
}