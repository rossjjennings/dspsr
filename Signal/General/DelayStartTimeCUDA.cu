//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2019 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DelayStartTimeCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

template<typename T>
__global__ void delay_fpt_kernel (const T * input, unsigned input_span,
                                  T * output, unsigned output_span,
                                  int64_t delay, uint64_t ndat)
{
  uint64_t idat = threadIdx.x + delay;
  uint64_t odat = threadIdx.x;

  //                        ichan      * npol      + ipol
  const unsigned ichanpol = blockIdx.y * gridDim.z + blockIdx.z;

  // offsets for channel and polarisation
  const T * in  = input + (ichanpol * input_span);
  T * out = output + (ichanpol * output_span);

  while (idat < ndat)
  {
    // delay the output by the specified amount
    out[odat] = in[idat];

    odat += blockDim.x;
    idat += blockDim.x;
  }
}

CUDA::DelayStartTimeEngine::DelayStartTimeEngine (cudaStream_t _stream)
{
  stream = _stream;
  delay_npol = 0;
  delay_nchan = 0;
  zero_delay = 0;
  d_delays = NULL;
}

CUDA::DelayStartTimeEngine::~DelayStartTimeEngine ()
{
  if (d_delays)
  {
    cudaError_t error = cudaFree (d_delays);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::DelayStartTimeEngine::~DelayStartTimeEngine", 
                   "cudaFree(%x): %s", (void *) &d_delays,
                   cudaGetErrorString (error));
  }
  d_delays = NULL;
}

void CUDA::DelayStartTimeEngine::delay(const dsp::TimeSeries* input, 
					                           dsp::TimeSeries* output,
                                     uint64_t output_ndat, int64_t delay)
{
  unsigned nchan = input->get_nchan ();
  unsigned npol = input->get_npol ();
  unsigned ndim = input->get_ndim ();

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
    cerr << "CUDA::DelayStartTimeEngine::delay ndim=" << ndim
         << " output_ndat=" << output_ndat 
         << " input.base=" << input_base
         << " input.span=" << input_span
         << " output.base=" << output_base 
         << " output.span=" << output_span
         << endl;

  if (output_ndat == 0)
    return;

  unsigned nthreads = 1024;
  dim3 blocks (1, nchan, npol);
  if (ndim == 1)
  {
    delay_fpt_kernel<float><<<blocks,nthreads,0,stream>>>((const float *) input_base, input_span,
                                                          (float *) output_base, output_span,
                                                          delay, output_ndat);
  }
  else if (ndim == 2)
  {
    delay_fpt_kernel<float2><<<blocks,nthreads,0,stream>>>((const float2 *) input_base, input_span,
                                                           (float2 *) output_base, output_span,
                                                           delay, output_ndat);
  }
  else if (ndim == 4)
  {
    delay_fpt_kernel<float4><<<blocks,nthreads,0,stream>>>((const float4 *) input_base, input_span,
                                                           (float4 *) output_base, output_span,
                                                           delay, output_ndat);
  }
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::DelayStartTimeEngine::delay delay_fpt_kernel", stream);
}
