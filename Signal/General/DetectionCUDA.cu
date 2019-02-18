//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2010 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/DetectionCUDA.h"

#include "Error.h"
#include "cross_detect.h"
#include "stokes_detect.h"
#include "templates.h"
#include "debug.h"

#include <memory>

#include <string.h>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

/*
  PP   = p^* p
  QQ   = q^* q
  RePQ = Re[p^* q]
  ImPQ = Im[p^* q]
*/

// number of input polarisations required for coherence
#define COHERENCE_NPOL 2

#define COHERENCE(PP,QQ,RePQ,ImPQ,p,q) \
  PP   = (p.x * p.x) + (p.y * p.y); \
  QQ   = (q.x * q.x) + (q.y * q.y); \
  RePQ = (p.x * q.x) + (p.y * q.y); \
  ImPQ = (p.x * q.y) - (p.y * q.x);


#define COHERENCE4(r,p,q) COHERENCE(r.w,r.x,r.y,r.z,p,q)

__global__ void coherence4 (const float2* input_base, uint64_t input_span,
                            float4 * output_base, uint64_t output_span,
                            uint64_t ndat)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ndat)
    return;

  const uint64_t input_offset = (blockIdx.y * input_span * COHERENCE_NPOL) + i;

  // read the input polarisations
  const float2 p = input_base[input_offset];
  const float2 q = input_base[input_offset + input_span];

  float4 r;

  COHERENCE4(r,p,q);

  output_base[(blockIdx.y * output_span) + i] = r;
}

#define COHERENCE2(s0,s1,p,q) COHERENCE(s0.x,s0.y,s1.x,s1.y,p,q)

__global__ void coherence2 (const float2* input_base, unsigned input_span, 
                            float2* output_base, unsigned output_span,
                            unsigned ndat)
{
  const float2* p0 = input_base + blockIdx.y * input_span * COHERENCE_NPOL;
  const float2* p1 = p0 + input_span;

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndat)
    return;

  float2 s0, s1;

  COHERENCE2(s0,s1,p0[i],p1[i]);

  float2* op0 = output_base + blockIdx.y * output_span * COHERENCE_NPOL;
  float2* op1 = op0 + output_span;

  op0[i] = s0;
  op1[i] = s1;
}

/*
  The input data are pairs of arrays of ndat complex numbers:
  The output are 4 pol, one for each coherency product
*/
__global__ void coherence1 (const float2* input_base, unsigned input_span,
                            float* output_base, unsigned output_span,
                            unsigned ndat)
{
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= ndat)
    return;

  //                             ichan      * input_span * npol
  const uint64_t input_offset = (blockIdx.y * input_span * COHERENCE_NPOL) + i;

  // read the input polarisations
  const float2 p = input_base[input_offset];
  const float2 q = input_base[input_offset + input_span];

  // PP
  //                        ichan      * pol_stride * npol + idat
  uint64_t output_offset = (blockIdx.y * output_span * 4) + i;
  output_base[output_offset] = (p.x * p.x) + (p.y * p.y);

  // QQ
  output_offset += output_span;
  output_base[output_offset] = (q.x * q.x) + (q.y * q.y);

  // Re(PQ)
  output_offset += output_span;
  output_base[output_offset] = (p.x * q.x) + (p.y * q.y);

  // Im(PQ)
  output_offset += output_span;
  output_base[output_offset] = (p.x * q.y) - (p.y * q.x);
}

CUDA::DetectionEngine::DetectionEngine (cudaStream_t _stream)
{
  stream = _stream;
}

void CUDA::DetectionEngine::polarimetry (unsigned ndim,
					 const dsp::TimeSeries* input, 
					 dsp::TimeSeries* output)
{
  if (ndim != 1 && ndim != 2 && ndim != 4)
    throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry",
		 "cannot handle ndim=%u != 1,2 or 4", ndim);

  uint64_t ndat = input->get_ndat ();
  unsigned nchan = input->get_nchan ();
  unsigned npol = input->get_npol ();

  if (npol != 2)
    throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry",
                 "input npol=%u != 2", npol);

  if (ndat != output->get_ndat ())
    throw Error (InvalidParam, "CUDA::DetectionEngine::polarimetry",
                 "input ndat=%u != output ndat=%u",
                 ndat, output->get_ndat());

  // all kernel inputs a npol=2, ndim=2
  const float2* input_base = (const float2*) input->get_datptr (0, 0);
  const float2* input_next = (const float2*) input->get_datptr (0, 1);
  uint64_t input_span = input_next - input_base;

  if (dsp::Operation::verbose)
    cerr << "CUDA::DetectionEngine::polarimetry ndim=" << output->get_ndim () 
         << " ndat=" << ndat 
         << " input.base=" << input_base
         << " input.span=" << input_span  << endl;

  if (ndat == 0)
    return;

  dim3 threads (128);
  dim3 blocks (ndat/threads.x, nchan);
  if (ndat % threads.x)
    blocks.x ++;

  uint64_t output_span = 0;
  // handle the different detection types
  if (ndim == 1)
  {
    float * output_base = output->get_datptr (0, 0);
    float * output_next = output->get_datptr (0, 1);
    output_span = output_next - output_base;

    if (dsp::Operation::verbose)
      cerr << "CUDA::DetectionEngine::polarimetry coherence1 output_base="
           << (void *) output_base << " output_span=" << output_span << endl;

    // output pol=4, dim=1 (out-of-place)
    coherence1<<<blocks,threads,0,stream>>>(input_base, input_span,
                                            output_base, output_span, ndat);
  }
  else if (ndim == 2)
  {
    float2 * output_base = (float2 *) output->get_datptr (0, 0);
    float2 * output_next = (float2 *) output->get_datptr (0, 1);
    output_span = output_next - output_base;

    if (dsp::Operation::verbose)
      cerr << "CUDA::DetectionEngine::polarimetry coherence2 output_base="
           << (void *) output_base << " output_span=" << output_span << endl;

    // output pol=2, dim=2 (in-place)
    coherence2<<<blocks,threads,0,stream>>>(input_base, input_span,
                                            output_base, output_span, ndat); 
  }
  else if (ndim == 4)
  {
    float4 * output_base = (float4 *) output->get_datptr (0, 0);
    if (nchan > 1)
    {
      float4 * output_next = (float4 *) output->get_datptr (1, 0);
      output_span = output_next - output_base;
    }

    if (dsp::Operation::verbose)
      cerr << "CUDA::DetectionEngine::polarimetry coherence4 output_base="
           << (void *) output_base << " output_span=" << output_span << endl;

    // output pol=1, dim=4 (out-of-place)
    coherence4<<<blocks,threads,0,stream>>> (input_base, input_span,
                                             output_base, output_span, ndat);
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream ("CUDA::DetectionEngine::polarimetry coherence", stream);
}

// dubiuous about the correctness here... TODO AJ
__global__ void sqld_tfp (float2 *base_in, unsigned stride_in,
                          float * base_out, unsigned stride_out, unsigned ndat)
{
  // input and output pointers for channel (y dim)
  float2 * in = base_in + (blockIdx.y * stride_in);
  float * out = base_out + (blockIdx.y * stride_out);

  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ndat)
    out[i] = in[i].x * in[i].x + in[i].y * in[i].y;
}

// form total intensity from input polarisations
__global__ void sqld_fpt (float2 *base_in, float *base_out, uint64_t ndat, 
                          unsigned npol, uint64_t in_stride, uint64_t out_stride)
{
  // set base pointer for ichan [blockIdx.y], ipol ==  0, input complex, output detected
  float2 * in = base_in + (blockIdx.y * npol * in_stride);
  float * out = base_out + (blockIdx.y * out_stride);

  // the idat
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idat < ndat)
  {
    float sum = 0.0f;
    uint64_t idx = idat;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const float2 v = in[idx];
      sum += v.x * v.x + v.y * v.y;
      idx += in_stride;
    }
    out[idat] = sum;
  }
}

// form PP,QQ output from dual pol input
__global__ void sqld_fpt_ppqq (float2 *base_in, float *base_out, uint64_t ndat,
                               uint64_t in_stride, uint64_t out_stride)
{
  //                         ichan        npol         ipol
  const unsigned ichanpol = (blockIdx.y * gridDim.z) + blockIdx.z;

  float2 * in = base_in  + (ichanpol * in_stride);
  float * out = base_out + (ichanpol * out_stride);
  const uint64_t idat = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idat < ndat)
  {
    const float2 v = in[idat];
    out[idat] = (v.x * v.x) + (v.y * v.y);
  }
}


void CUDA::DetectionEngine::square_law (const dsp::TimeSeries* input,
           dsp::TimeSeries* output)
{
  uint64_t ndat  = input->get_ndat ();
  unsigned nchan = input->get_nchan ();
  unsigned ndim  = input->get_ndim();
  unsigned npol  = input->get_npol();

  unsigned output_npol = output->get_npol();

  if (ndim != 2)
    throw Error (InvalidParam, "CUDA::DetectionEngine::square_law",
     "cannot handle ndim=%u != 2", ndim);

  if (npol == 1 && output_npol == 2)
    throw Error (InvalidParam, "CUDA::DetectionEngine::square_law",
      "cannot form PPQQ output from single polarisation input");

  if (input == output && output_npol == 1)
    throw Error (InvalidParam, "CUDA::DetectionEngine::square_law"
     "cannot handle in-place data for Intensity output");

  switch (input->get_order())
  {
    case dsp::TimeSeries::OrderTFP:
    {
      dim3 threads (512);
      dim3 blocks (ndat/threads.x, nchan);

      if (ndat % threads.x)
        blocks.x ++;

      float2* base_in = (float2*) input->get_dattfp ();
      float* base_out = output->get_dattfp();

      unsigned stride_in = nchan * npol;
      unsigned stride_out = nchan * npol;

      if (dsp::Operation::verbose)
        cerr << "CUDA::DetectionEngine::square_law sqld_tfp ndat=" << ndat
             << " stride_in=" << stride_in << " stride_out=" << stride_out << endl;

      sqld_tfp<<<blocks,threads,0,stream>>> (base_in, stride_in, base_out, stride_out, ndat);

      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream ("CUDA::DetectionEngine::square_law sqld_tfp", stream);

      break;
    }

    case dsp::TimeSeries::OrderFPT:
    {
      float2* base_in = (float2*) input->get_datptr (0, 0);
      float* base_out = output->get_datptr(0, 0);

      uint64_t in_stride = ndat;
      uint64_t out_stride = ndat;

      // determine the stride between blocks on ndat
      if ((npol == 1) && (nchan > 1))
      {
        float2 * next = (float2*) input->get_datptr (1, 0);
        in_stride = next - base_in;
      }
      else if (npol == 2)
      {
        float2 * next = (float2*) input->get_datptr (0, 1);
        in_stride = next - base_in;
      }
      else
        throw Error (InvalidState, "CUDA::DetectionEngine::square_law",
                     "unsupported combination of input npol=%u and nchan=%u", npol, nchan);

      if (output_npol == 2)
      {
        float * next = (float*) output->get_datptr (0, 1);
        out_stride = next - base_out;
      }
      else if (nchan > 1)
      {
        float * next = (float*) output->get_datptr (1, 0);
        out_stride = next - base_out;
      }
      else
        throw Error (InvalidState, "CUDA::DetectionEngine::square_law",
                     "unsupported combination of output npol=%u and nchan=%u", output_npol, nchan);

      if (dsp::Operation::verbose)
        cerr << "CUDA::DetectionEngine::square_law sqld_fpt "
             << " base_in=" << (void *) base_in
             << " base_out=" << (void *) base_out
             << " ndat=" << ndat << " nchan=" << nchan 
             << " npol=" << npol<< " in_stride=" << in_stride 
             << " out_stride=" << out_stride << endl;

      dim3 threads (512);
      if (output_npol == 2)
      {
        dim3 blocks (ndat/threads.x, nchan, npol);
        if (ndat % threads.x)
          blocks.x ++;

        sqld_fpt_ppqq<<<blocks,threads,0,stream>>> (base_in, base_out, ndat, in_stride, out_stride);

        if (dsp::Operation::record_time || dsp::Operation::verbose)
          check_error_stream ("CUDA::DetectionEngine::square_law sqld_fpt_ppqq", stream);
      }
      else
      {
        dim3 blocks (ndat/threads.x, nchan);
        if (ndat % threads.x)
          blocks.x ++;

        sqld_fpt<<<blocks,threads,0,stream>>> (base_in, base_out, ndat, npol, in_stride, out_stride);

        if (dsp::Operation::record_time || dsp::Operation::verbose)
          check_error_stream ("CUDA::DetectionEngine::square_law sqld_fpt", stream);
      }

      break;
    }

    default:
    {
      throw Error (InvalidState, "CUDA::DetectionEngine::square_law", "unrecognized order");
    }
  }
}
