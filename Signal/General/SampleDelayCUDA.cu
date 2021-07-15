//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/SampleDelayCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

template<typename T>
__global__ void retard_fpt_kernel (const T * input, unsigned input_span,
                                   T * output, unsigned output_span,
                                   int64_t * delays, uint64_t ndat)
{
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ndat)
    return;

  //                        ichan      * npol      + ipol
  const unsigned ichanpol = blockIdx.y * gridDim.z + blockIdx.z;

  // offsets for input and output
  const uint64_t idx = (ichanpol * input_span) + i + delays[ichanpol];
  const uint64_t odx = (ichanpol * output_span) + i;

  // delay the output by the specified amount
  output[odx] = input[idx];
}

CUDA::SampleDelayEngine::SampleDelayEngine (cudaStream_t _stream)
{
  stream = _stream;
  delay_npol = 0;
  delay_nchan = 0;
  d_delays = NULL;
}

CUDA::SampleDelayEngine::~SampleDelayEngine ()
{
  if (d_delays)
  {
    cudaError_t error = cudaFree (d_delays);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::SampleDelayEngine::~SampleDelayEngine",
                   "cudaFree(%x): %s", (void *) &d_delays,
                   cudaGetErrorString (error));
  }
  d_delays = NULL;
}

void CUDA::SampleDelayEngine::set_delays (unsigned npol, unsigned nchan,
                                          vector<int64_t> zero_delays,
                                          dsp::SampleDelayFunction * function)
{
  delay_npol = npol;
  delay_nchan = nchan;
  delays_size = npol * nchan * sizeof(int64_t);

  cudaError_t error;

  // ensure delays arrays are appropriately sized
  if (d_delays)
  {
    error = cudaFree (d_delays);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::SampleDelayEngine::set_delays",
                   "cudaFree(%xu): %s", (void *) &d_delays,
                   cudaGetErrorString (error));
  }
  d_delays = NULL;
  error = cudaMalloc((void **) &d_delays, delays_size);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::SampleDelayEngine::set_delays",
                 "cudaMalloc(ptr, %ld): %s", delays_size,
                 cudaGetErrorString (error));

  // temporary array for transfer to device
  int64_t * h_delays = NULL;
  error = cudaMallocHost((void **) &h_delays, delays_size);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::SampleDelayEngine::set_delays",
                 "cudaMallocHost(ptr, %ld): %s", delays_size,
                 cudaGetErrorString (error));

  for (unsigned ichan=0; ichan<delay_nchan; ichan++)
  {
    for (unsigned ipol=0; ipol<delay_npol; ipol++)
    {
      int64_t applied_delay = 0;
      if (zero_delays[ichan])
        applied_delay = zero_delays[ichan] - function->get_delay(ichan, ipol);
      else
        applied_delay = function->get_delay(ichan, ipol);
      if (applied_delay < 0)
        throw Error (InvalidState, "CUDA::SampleDelayEngine::set_delays",
                     "delay for ipol=%u ichan=%u was %ld which must not be negative",
                     ipol, ichan, applied_delay);
      h_delays[ichan * npol + ipol] = applied_delay;
    }
  }

  // transfer host to device, and wait for the copy to complete
  error = cudaMemcpyAsync ((void *)d_delays, (void *)h_delays,
                           delays_size, cudaMemcpyHostToDevice, stream);
  if (error != cudaSuccess)
    throw Error (FailedCall, "CUDA::SampleDelayEngine::set_delays",
                 "cudaMemcpyAsync(%xu,%xu,%ld): %s", (void *) d_delays,
                  (void *)h_delays, delays_size, cudaGetErrorString (error));

  cudaStreamSynchronize(stream);

  // don't retain the host delays
  error = cudaFreeHost (h_delays);
  if (error != cudaSuccess)
     throw Error (FailedCall, "CUDA::SampleDelayEngine::set_delays",
                 "cudaFreeHost(%xu): %s", (void *) &h_delays,
                 cudaGetErrorString (error));
}

void CUDA::SampleDelayEngine::retard(const dsp::TimeSeries* input,
					                           dsp::TimeSeries* output,
                                     uint64_t output_ndat)
{
  unsigned nchan = input->get_nchan ();
  unsigned npol = input->get_npol ();
  unsigned ndim = input->get_ndim ();

  if (npol != delay_npol || nchan != delay_nchan)
    throw Error (InvalidState, "CUDA::SampleDelayEngine::retard",
                 "delay dim [%u,%u] did not match input dim [%u,%u]",
                  delay_nchan, delay_npol, nchan, npol);

  const float * input_base = input->get_datptr (0, 0);
  float * output_base = output->get_datptr (0, 0);
  uint64_t input_span, output_span;

  if (npol > 1)
  {
    input_span = (input->get_datptr (0, 1) - input_base) / ndim;
    output_span = (output->get_datptr (0, 1) - output_base) / ndim;
  }
  else if (npol == 1 && nchan > 1)
  {
    input_span = (input->get_datptr (1, 0) - input_base) / ndim;
    output_span = (output->get_datptr (1, 0) - output_base) / ndim;
  }
  else
  {
    input_span = output_span = 0;
  }

  if (dsp::Operation::verbose)
    cerr << "CUDA::SampleDelayEngine::retard ndim=" << ndim
         << " output_ndat=" << output_ndat
         << " input.base=" << input_base
         << " input.span=" << input_span
         << " output.base=" << output_base
         << " output.span=" << output_span
         << endl;

  if (output_ndat == 0)
    return;

  unsigned nthreads = 1024;
  dim3 blocks (output_ndat / nthreads, nchan, npol);
  if (output_ndat % nthreads != 0)
    blocks.x ++;

  if (ndim == 1)
  {
    retard_fpt_kernel<float><<<blocks,nthreads,0,stream>>>((const float *) input_base, input_span,
                                                           (float *) output_base, output_span,
                                                           d_delays, output_ndat);
  }
  else if (ndim == 2)
  {
    retard_fpt_kernel<float2><<<blocks,nthreads,0,stream>>>((const float2 *) input_base, input_span,
                                                            (float2 *) output_base, output_span,
                                                            d_delays, output_ndat);
  }
  else if (ndim == 4)
  {
    retard_fpt_kernel<float4><<<blocks,nthreads,0,stream>>>((const float4 *) input_base, input_span,
                                                            (float4 *) output_base, output_span,
                                                            d_delays, output_ndat);
  }
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::SampleDelayEngine::retard retard_fpt_kernel", stream);
}
