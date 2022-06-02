//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/PScrunchCUDA.h"

#include "Error.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

void check_error (const char*);

CUDA::PScrunchEngine::PScrunchEngine (cudaStream_t _stream)
{
  stream = _stream;
}

CUDA::PScrunchEngine::~PScrunchEngine ()
{
}

//! get cuda device properties
void CUDA::PScrunchEngine::setup()
{
  gpu_config.init();
}


//
// each thread reads a single value from both polarisation
// and adds them together
//
__global__ void fpt_pscrunch_intensity_kernel (const float * __restrict__ in_p0,
                                               const float * __restrict__ in_p1,
                                               float * out, uint64_t in_chan_span,
                                               uint64_t out_chan_span, uint64_t in_ndat)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= in_ndat)
    return;

  // increment the input/output base pointers to this chan/pol
  in_p0 += (blockIdx.y * in_chan_span);
  in_p1 += (blockIdx.y * in_chan_span);
  out   += (blockIdx.y * out_chan_span);

  out[idx] = in_p0[idx] + in_p1[idx];
}

__global__ void fpt_pscrunch_p_state_kernel (const float * __restrict__ in_p,
                                             float * out, uint64_t in_chan_span,
                                             uint64_t out_chan_span, uint64_t in_ndat)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= in_ndat)
    return;

  // increment the input/output base pointers to this chan/pol
  in_p += (blockIdx.y * in_chan_span);
  out  += (blockIdx.y * out_chan_span);

  out[idx] = in_p[idx];
}

__global__ void fpt_pscrunch_pq_state_kernel (const float * __restrict__ in_p0,
                                              const float * __restrict__ in_p1,
                                              float * out_p0, float * out_p1,
                                              uint64_t in_chan_span,
                                              uint64_t out_chan_span, uint64_t in_ndat)
{
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= in_ndat)
    return;

  // increment the input/output base pointers to this chan/pol
  in_p0 += (blockIdx.y * in_chan_span);
  in_p1 += (blockIdx.y * in_chan_span);
  out_p0 += (blockIdx.y * out_chan_span);
  out_p1 += (blockIdx.y * out_chan_span);

  out_p0[idx] = in_p0[idx];
  out_p1[idx] = in_p1[idx];
}

void CUDA::PScrunchEngine::fpt_pscrunch (const dsp::TimeSeries* input, 
                                         dsp::TimeSeries* output)
{
  if (input == output)
    throw Error (InvalidParam, "CUDA::PScrunchEngine::fpt_pscrunch",
     "cannot handle in-place data");

  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol  = input->get_npol();

  uint64_t in_chan_span = 0;
  uint64_t out_chan_span = 0;
  if (input_nchan > 1)
  {
    in_chan_span = input->get_datptr (1, 0) - input->get_datptr (0, 0);
    out_chan_span = output->get_datptr (1, 0) - output->get_datptr (0, 0);
  }

#ifdef _DEBUG
  cerr << "CUDA::PScrunchEngine::fpt_pscrunch channel spans: input=" << in_chan_span << " output=" << out_chan_span << endl;
#endif

  dim3 threads (gpu_config.get_max_threads_per_block());
  dim3 blocks (input_ndat / threads.x, input_nchan);
  if (input_ndat % threads.x)
    blocks.x ++;

  const float * in_pol0 = input->get_datptr (0, 0);
  const float * in_pol1 = input->get_datptr (0, 1);

  if (output->get_npol() == 1)
  {
    float * out = output->get_datptr (0, 0);
    if (output->get_state() == Signal::Intensity)
    {
      if (dsp::Operation::verbose)
        cerr << "CUDA::PScrunchEngine::fpt_pscrunch fpt_pscrunch_intensity_kernel" << endl;
      fpt_pscrunch_intensity_kernel<<<blocks,threads,0,stream>>> (in_pol0, in_pol1, out, in_chan_span, out_chan_span, input_ndat);
    }
    else if (output->get_state() == Signal::PP_State)
    {
      if (dsp::Operation::verbose)
        cerr << "CUDA::PScrunchEngine::fpt_pscrunch fpt_pscrunch_p_state_kernel" << endl;
      fpt_pscrunch_p_state_kernel<<<blocks,threads,0,stream>>> (in_pol0, out, in_chan_span, out_chan_span, input_ndat);
    }
    else if (output->get_state() == Signal::QQ_State)
    {
      if (dsp::Operation::verbose)
        cerr << "CUDA::PScrunchEngine::fpt_pscrunch fpt_pscrunch_p_state_kernel" << endl;
      fpt_pscrunch_p_state_kernel<<<blocks,threads,0,stream>>> (in_pol1, out, in_chan_span, out_chan_span, input_ndat);
    }
  }
  else if (output->get_npol() == 2)
  {
    float * out_p0 = output->get_datptr (0, 0);
    float * out_p1 = output->get_datptr (0, 1);
    if (dsp::Operation::verbose)
      cerr << "CUDA::PScrunchEngine::fpt_pscrunch fpt_pscrunch_pq_state_kernel" << endl;
    fpt_pscrunch_pq_state_kernel<<<blocks,threads,0,stream>>> (in_pol0, in_pol1, out_p0, out_p1, in_chan_span, out_chan_span, input_ndat);
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error ("CUDA::PScrunchEngine::fpt_pscrunch");
}


// each block pscrunches 1 time sample for many channels
template<typename T>
__global__ void tfp_pscrunch_intensity_kernel (const T * __restrict__ in, float * out, unsigned nchan)
{
  const unsigned ichan = blockIdx.x * blockDim.x + threadIdx.x;
  if (ichan >= nchan)
    return;

  const unsigned idx = blockIdx.y * nchan + ichan;
  out[idx] = in[idx].x + in[idx].y;
}

template<typename T>
__global__ void tfp_pscrunch_pp_state_kernel (const T * __restrict__ in, float * out, unsigned nchan)
{
  const unsigned ichan = blockIdx.x * blockDim.x + threadIdx.x;
  if (ichan >= nchan)
    return;

  const unsigned idx = blockIdx.y * nchan + ichan;
  out[idx] = in[idx].x;
}

template<typename T>
__global__ void tfp_pscrunch_qq_state_kernel (const T * __restrict__ in, float * out, unsigned nchan)
{
  const unsigned ichan = blockIdx.x * blockDim.x + threadIdx.x;
  if (ichan >= nchan)
    return;

  const unsigned idx = blockIdx.y * nchan + ichan;
  out[idx] = in[idx].y;
}

template<typename T>
__global__ void tfp_pscrunch_pq_state_kernel (const T * __restrict__ in, float2 * out, unsigned nchan)
{
  const unsigned ichan = blockIdx.x * blockDim.x + threadIdx.x;
  if (ichan >= nchan)
    return;

  const unsigned idx = blockIdx.y * nchan + ichan;
  out[idx] = make_float2(in[idx].x, in[idx].y);
}

void CUDA::PScrunchEngine::tfp_pscrunch (const dsp::TimeSeries* input,
                                         dsp::TimeSeries* output)
{
  if (input == output)
    throw Error (InvalidParam, "CUDA::PScrunchEngine::tfp_pscrunch"
     "cannot handle in-place data");

  if (input->get_npol() == 2)
    tfp_pscrunch_launch<float2>(input, output);
  else if (input->get_npol() == 4)
    tfp_pscrunch_launch<float4>(input, output);
  else
    throw Error(InvalidParam, "CUDA::PScrunchEngine::tfp_scrunch", "incompatible combintion of input and output npol");
}

template<typename S>
void CUDA::PScrunchEngine::tfp_pscrunch_launch(const dsp::TimeSeries* input, dsp::TimeSeries* output)
{
  const uint64_t input_ndat  = input->get_ndat();
  const unsigned input_nchan = input->get_nchan();
  const unsigned input_npol  = input->get_npol();
  const unsigned output_npol = output->get_npol();
  const Signal::State state  = output->get_state();

  dim3 threads (gpu_config.get_max_threads_per_block());
  dim3 blocks (input_nchan*input_npol/threads.x, input_ndat);
  if (input_nchan*input_npol % threads.x)
    blocks.x++;

  const S * in = (S *) input->get_dattfp ();
  if (output_npol == 1)
  {
    float * out = (float *) output->get_dattfp ();
    if (state == Signal::Intensity)
      tfp_pscrunch_intensity_kernel<S><<<blocks,threads,0,stream>>> (in, out, input_nchan);
    else if (state == Signal::PP_State)
      tfp_pscrunch_pp_state_kernel<S><<<blocks,threads,0,stream>>> (in, out, input_nchan);
    else if (state == Signal::QQ_State)
      tfp_pscrunch_qq_state_kernel<S><<<blocks,threads,0,stream>>> (in, out, input_nchan);
  }
  else if (output_npol == 2)
  {
    float2 * out = (float2 *) output->get_dattfp ();
    tfp_pscrunch_pq_state_kernel<S><<<blocks,threads,0,stream>>> (in, out, input_nchan);
  }
}
