//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2015 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ConvolutionCUDASpectral.h"
#include "CUFFTError.h"
#include "debug.h"

#if HAVE_CUFFT_CALLBACKS
#include "dsp/ConvolutionCUDACallbacks.h"
#include <cufftXt.h>
#endif

#include <iostream>
#include <cassert>

using namespace std;

void check_error_stream (const char*, cudaStream_t);

// multiply by the dedispersion kernel for all polarisations
// ichan  == blockIdx.y
// ipt_bwd == blockIdx.x * blockDim.x + threadIdx.x
__global__ void k_multiply_conv_spectral (float2* d_fft, const __restrict__ float2 * kernel, unsigned npt_bwd, unsigned npol)
{
  //              ichan * chan_stride   + (ipt)
  uint64_t idx = (blockIdx.y * npt_bwd) + (blockIdx.x * blockDim.x + threadIdx.x);

  // load the dedispersion kernel
  const float2 kern = kernel[idx];
  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    d_fft[idx] = cuCmulf(d_fft[idx], kern);
    idx += gridDim.y * npt_bwd;
  }
}

// ichan == blockIdx.y
// ipt_bwd == blockIdx.x * blockDim.x + threadIdx.x
__global__ void k_ncopy_conv_spectral_pft (float2* output_data, uint64_t opol_stride,
           const __restrict__ float2* input_data, unsigned ipol_stride,
           unsigned nfilt_pos, unsigned nsamp_step, unsigned npol)
{
  const unsigned ipt_bwd = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (ipt_bwd < nfilt_pos)
    return;

  const unsigned ichan = blockIdx.y;
  const unsigned osamp = ipt_bwd - nfilt_pos;
  if (osamp >= nsamp_step)
    return;

  uint64_t idx = (ichan * ipol_stride)        + ipt_bwd;
  uint64_t odx = (ichan * opol_stride * npol) + osamp;

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    output_data[odx] = input_data[idx];

    idx += gridDim.y * ipol_stride;
    odx += opol_stride;
  }
}

__global__ void k_ncopy_conv_spectral_fpt (float2* output_data, uint64_t opol_stride,
           const __restrict__ float2* input_data, unsigned ipol_stride,
           unsigned nfilt_pos, unsigned nsamp_step, unsigned npol)
{
  const unsigned ipt = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (ipt < nfilt_pos)
    return;

  const unsigned ichan = blockIdx.y;
  const unsigned osamp = ipt - nfilt_pos;
  if (osamp >= nsamp_step)
    return;

  uint64_t idx = (ichan * npol * ipol_stride) + ipt;
  uint64_t odx = (ichan * npol * opol_stride) + osamp;

  for (unsigned ipol=0; ipol<npol; ipol++)
  {
    output_data[odx] = input_data[idx];

    idx += ipol_stride;
    odx += opol_stride;
  }
}

CUDA::ConvolutionEngineSpectral::ConvolutionEngineSpectral (cudaStream_t _stream)
{
  stream = _stream;

  // create plan handles
  cufftResult result;

  result = cufftCreate (&plan_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::ConvolutionEngineSpectral",
                      "cufftCreate(plan_fwd)");

  result = cufftCreate (&plan_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::ConvolutionEngineSpectral",
                      "cufftCreate(plan_bwd)");

  fft_configured = false;
  nchan = 0;
  npt_fwd = 0;
  npt_bwd = 0;

  work_area = 0;
  work_area_size = 0;

  buf = 0;
  buf_size = 0;
  d_kernels = 0;

  input_stride = 0;
  output_stride = 0;

#ifdef HAVE_CUFFT_CALLBACKS
  h_conv_params_size = sizeof(unsigned) * 3;
  cudaError_t error = cudaMallocHost ((void **) &h_conv_params, h_conv_params_size);
  if (error != cudaSuccess)
    throw Error (InvalidState, "CUDA::ConvolutionEngineSpectralCallbacks::ConvolutionEngineSpectralCallbacks",
                 "could not allocate memory for h_conv_params");
#endif
}

CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral()
{
  cufftResult result;

  result = cufftDestroy (plan_fwd);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral "
         << "failed to destroy plan_fwd" << endl;

  result = cufftDestroy (plan_bwd);
  if (result != CUFFT_SUCCESS)
    cerr << "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral "
         << "failed to destroy plan_bwd" << endl;

  if (work_area)
  {
    cudaError_t error = cudaFree (work_area);
    if (error != cudaSuccess)
      cerr << "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral "
           << "failed to free work_area " << cudaGetErrorString (error) << endl;
  }

  if (buf)
  {
    cudaError_t error = cudaFree (buf);
    if (error != cudaSuccess)
      cerr << "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral "
           << "failed to free buf: " << cudaGetErrorString (error) << endl;
  }

#if HAVE_CUFFT_CALLBACKS
  if (h_conv_params)
  {
    cudaError_t error = cudaFreeHost (h_conv_params);
    if (error != cudaSuccess)
      cerr << "CUDA::ConvolutionEngineSpectral::~ConvolutionEngineSpectral "
           << "failed to free h_conv_paramsf: " << cudaGetErrorString (error) << endl;
  }
#endif
}

void CUDA::ConvolutionEngineSpectral::regenerate_plans()
{
  cufftDestroy (plan_fwd);
  cufftCreate (&plan_fwd);
  cufftDestroy (plan_bwd);
  cufftCreate (&plan_bwd);
}

void CUDA::ConvolutionEngineSpectral::set_scratch (void * scratch)
{
  d_scratch = (cufftComplex *) scratch;
}

// prepare all relevant attributes for the engine
void CUDA::ConvolutionEngineSpectral::prepare (dsp::Convolution * convolution)
{
  const dsp::Response* response = convolution->get_response();

  nchan = response->get_nchan();
  npt_bwd = response->get_ndat();
  npt_fwd = convolution->get_minimum_samples();
  nsamp_overlap = convolution->get_minimum_samples_lost();
  nsamp_step = npt_fwd - nsamp_overlap;
  nfilt_pos = response->get_impulse_pos ();
  nfilt_neg = response->get_impulse_neg ();

  if (convolution->get_input()->get_state() == Signal::Nyquist)
    type_fwd = CUFFT_R2C;
  else
    type_fwd = CUFFT_C2C;

  // configure the dedispersion kernel
  setup_kernel (convolution->get_response());

  fft_configured = false;

  // initialize the kernel size configuration
  mp.init();
  mp.set_nelement (npt_bwd);
}

// setup the convolution kernel based on the reposnse
void CUDA::ConvolutionEngineSpectral::setup_kernel (const dsp::Response * response)
{
  unsigned nchan = response->get_nchan();
  unsigned ndat = response->get_ndat();
  unsigned ndim = response->get_ndim();

  assert (ndim == 2);
  assert (d_kernels == 0);

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::setup_kernel response: "
         << "nchan=" << nchan << " ndat=" << ndat << " ndim=" << ndim << endl;

  // allocate memory for dedispersion kernel of all channels
  unsigned kernels_size = ndat * sizeof(cufftComplex) * nchan;
  cudaError_t error = cudaMalloc ((void**)&d_kernels, kernels_size);
  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngineSpectral::setup_kernel",
     "could not allocate device memory for dedispersion kernel");
  }

  // copy all kernels from host to device
  const float* kernel = response->get_datptr (0,0);

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::setup_kernel cudaMemcpy stream="
         << stream << " size=" << kernels_size << endl;
  if (stream)
    error = cudaMemcpyAsync (d_kernels, kernel, kernels_size, cudaMemcpyHostToDevice, stream);
  else
    error = cudaMemcpy (d_kernels, kernel, kernels_size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
  {
    throw Error (InvalidState, "CUDA::ConvolutionEngineSpectral::setup_kernel",
     "could not copy dedispersion kernel to device");
  }

  cudaStreamSynchronize(stream);
}

// configure the batched FFT plans
void CUDA::ConvolutionEngineSpectral::setup_batched (const dsp::TimeSeries* input,
                                                     dsp::TimeSeries * output,
                                                     dsp::TimeSeries * output_zdm)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::setup_batched npt_fwd=" << npt_fwd
         << " npt_bwd=" << npt_bwd << endl;

  nchan = input->get_nchan();
  npol  = input->get_npol();
  unsigned ndim = input->get_ndim();

#ifdef _DEBUG
  cerr << "CUDA::ConvolutionEngineSpectral::setup_batched nchan=" << nchan
       << " npol=" << npol << " ndat=" << input->get_ndat() << endl;
#endif

  input_stride = (input->get_datptr (1, 0) - input->get_datptr (0, 0)) / ndim;
  output_stride = (output->get_datptr (1, 0) - output->get_datptr (0, 0) ) / ndim;
  if (output_zdm)
    output_zdm_stride = (output_zdm->get_datptr (1, 0) - output_zdm->get_datptr (0, 0) ) / ndim;
  else
    output_zdm_stride = 0;
  buf_stride = npt_bwd;

  int rank = 1;
  int inembed[1];
  int onembed[1];
  int istride, ostride, idist, odist;
  cufftResult result;

  // now setup the forward batched plan
  size_t work_size_fwd, work_size_bwd;

  // complex layout plans for input
  inembed[0] = npt_fwd;
  onembed[0] = npt_bwd;

  // distance between fft samples
  istride = 1;
  ostride = 1;

  // distance between fft blocks
  idist = (int) input_stride;
  odist = (int) buf_stride;

#ifdef _DEBUG
  cerr << "CUDA::ConvolutionEngineSpectral::setup_batched npt_fwd=" << npt_fwd
       << " nbatch=" << nchan << endl;
  cerr << "CUDA::ConvolutionEngineSpectral::setup_batched input_stride="
       << input_stride << " output_stride=" << output_stride << endl;
#endif

  // setup forward fft
  result = cufftMakePlanMany (plan_fwd, rank, &npt_fwd,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              type_fwd, nchan, &work_size_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftMakePlanMany (plan_fwd)");

  result = cufftSetStream (plan_fwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
          "cufftSetStream(plan_fwd)");

  // get a rough estimate on work buffer size
  work_size_fwd = 0;
  result = cufftEstimateMany(rank, &npt_fwd,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             type_fwd, nchan, &work_size_fwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftEstimateMany(plan_fwd)");

  // distance between FFT samples
  istride = 1;
  ostride = 1;

  inembed[0] = npt_bwd;
  onembed[0] = npt_bwd;

  // distance between FFT blocks
  idist = (int) buf_stride;
#ifdef HAVE_CUFFT_CALLBACKS
  odist = (int) output_stride;
#else
  odist = (int) buf_stride;
#endif

  // the backward FFT is a has a simple layout (npt_bwd)
  DEBUG("CUDA::ConvolutionEngineSpectral::setup_batched cufftMakePlanMany (plan_bwd)");
  result = cufftMakePlanMany (plan_bwd, rank, &npt_bwd,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_C2C, nchan, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftMakePlanMany (plan_bwd)");

  result = cufftSetStream (plan_bwd, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftSetStream(plan_bwd)");

  DEBUG("CUDA::ConvolutionEngineSpectral::setup_batched bwd FFT plan set");

  work_size_bwd = 0;
  result = cufftEstimateMany(rank, &npt_bwd,
                             inembed, istride, idist,
                             onembed, ostride, odist,
                             CUFFT_C2C, nchan, &work_size_bwd);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::setup_batched",
                      "cufftEstimateMany(plan_fwd)");

  // free the space allocated for buf in setup_singular
  cudaError_t error;
#ifdef HAVE_CUFFT_CALLBACKS
  size_t batched_buffer_size = npt_bwd * nchan * sizeof (cufftComplex);
#else
  size_t batched_buffer_size = npt_bwd * nchan * npol * sizeof (cufftComplex);
#endif
  if (batched_buffer_size > buf_size)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::ConvolutionEngineSpectral::setup_batched batched_buffer_"
           << "size=" << batched_buffer_size << " buf_size=" << buf_size << endl;
    if (buf)
    {
      error = cudaFree (buf);
      if (error != cudaSuccess)
        throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup_batched",
                     "cudaFree(%x): %s", &buf, cudaGetErrorString (error));
    }

    error = cudaMalloc ((void **) &buf, batched_buffer_size);
    if (error != cudaSuccess)
      throw Error (FailedCall, "CUDA::ConvolutionEngineSpectral::setup_batched",
                   "cudaMalloc(%x, %u): %s", &buf, batched_buffer_size,
                   cudaGetErrorString (error));
    buf_size = batched_buffer_size;
  }

#ifdef HAVE_CUFFT_CALLBACKS
  // configured the CUFFT callbacks, associating them with the FFT plans
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::setup_batched "
         << "setup_callbacks_ConvolutionCUDASpectral()" << endl;
  setup_callbacks_ConvolutionCUDASpectral (plan_fwd, plan_bwd, d_kernels, stream);

  // configure the convolution parameters neeeded for the bwd fft stroe
  h_conv_params[0] = unsigned(output_stride);
  h_conv_params[1] = nfilt_pos;
  h_conv_params[2] = npt_bwd - nfilt_neg;
  setup_callbacks_conv_params_spectral (h_conv_params, h_conv_params_size, stream);
#endif

  fft_configured = true;
}

void CUDA::ConvolutionEngineSpectral::perform (
  const dsp::TimeSeries* input,
  dsp::TimeSeries * output,
  unsigned npart
)
{
  perform(input, output, NULL, npart);
}

// Perform convolution choosing the optimal batched size or if ndat is not as
// was configured, then perform singular
void CUDA::ConvolutionEngineSpectral::perform (const dsp::TimeSeries* input, dsp::TimeSeries * output, dsp::TimeSeries * output_zdm, unsigned npart)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform (" << npart << ")" << endl;

  if (npart == 0)
    return;

  uint64_t curr_istride = (input->get_datptr (1, 0) - input->get_datptr (0, 0)) / input->get_ndim();
  uint64_t curr_ostride = (output->get_datptr (1, 0) - output->get_datptr (0, 0)) / output->get_ndim();

  if (dsp::Operation::verbose)
  {
    cerr << "CUDA::ConvolutionEngineSpectral::perform istride prev=" << input_stride << " curr=" << curr_istride << " ndim=" << input->get_ndim() << endl;
    cerr << "CUDA::ConvolutionEngineSpectral::perform ostride prev=" << output_stride << " curr=" << curr_ostride << " ndim=" << output->get_ndim() << endl;
  }

  if (curr_istride != input_stride || curr_ostride != output_stride)
  {
    if (dsp::Operation::verbose)
      cerr << "CUDA::ConvolutionEngineSpectral::perform reconfiguring FFT batch sizes" << endl;
    fft_configured = false;
  }

  if (!fft_configured)
  {
    regenerate_plans ();
    setup_batched (input, output, output_zdm);
  }

  if (type_fwd == CUFFT_C2C)
  {
    perform_complex (input, output, output_zdm, npart);
  }
  else
  {
    cerr << "CUDA::ConvolutionEngineSpectral::perform_real not implemented" << endl;
    //perform_real (input, output, npart);
  }
}

void CUDA::ConvolutionEngineSpectral::perform_complex (const dsp::TimeSeries* input,
                                                       dsp::TimeSeries * output,
                                                       dsp::TimeSeries * output_zdm,
                                                       unsigned npart)
{
  const unsigned npol = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndim = input->get_ndim();
  const uint64_t ipol_stride = input_stride / npol;
  const uint64_t opol_stride = output_stride / npol;
  const uint64_t opol_zdm_stride = output_zdm_stride / npol;
  const uint64_t bpol_stride = nchan * npt_bwd;

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform_complex npart=" << npart
         << " nsamp_step=" << nsamp_step << endl;

  // forward FFT all the data for both polarisations (into FPT order)
  cufftComplex *  in_t = (cufftComplex *) input->get_datptr (0, 0);
  cufftComplex * buf_t = (cufftComplex *) buf;
  cufftComplex * out_t = (cufftComplex *) output->get_datptr (0, 0);
  cufftComplex * out_zdm_t = NULL;
  if (output_zdm != NULL)
    out_zdm_t = (cufftComplex *) output_zdm->get_datptr (0, 0);
  cufftResult result;

#ifndef HAVE_CUFFT_CALLBACKS
  dim3 blocks = dim3 (npt_bwd / mp.get_nthread(), nchan, 1);
  unsigned nthreads = mp.get_nthread();
  if (npt_bwd <= nthreads)
  {
    blocks.x = 1;
    nthreads = npt_bwd;
  }
  else
  {
    if (npt_bwd % nthreads)
      blocks.x++;
  }
#endif

  if (dsp::Operation::verbose)
    cerr << "CUDA::ConvolutionEngineSpectral::perform_complex in="
         << in_t << " buf=" << buf_t << " out_t=" << out_t
         << " out_zdm_t=" << out_zdm_t << endl;

  for (unsigned ipart=0; ipart<npart; ipart++)
  {
    if (output_zdm)
    {
      // perform the ncopy kernel for both polarisations from TFP to TFP
      k_ncopy_conv_spectral_fpt<<<blocks, nthreads, 0, stream>>> (out_zdm_t, opol_zdm_stride,
                                                                  in_t, ipol_stride,
                                                                  nfilt_pos, nsamp_step, npol);
    }

    cufftComplex * in_ptr  = in_t;
    cufftComplex * out_ptr = out_t;
    cufftComplex * buf_ptr = buf_t;

#ifdef HAVE_CUFFT_CALLBACKS
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      // forward fft + convolution with kernel
      result = cufftExecC2C (plan_fwd, in_ptr, buf_t, CUFFT_FORWARD);
      if (result != CUFFT_SUCCESS)
        throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_complex",
                          "cufftExecC2C(plan_fwd)");

      // backward fft + sample overlap
      result = cufftExecC2C (plan_bwd, buf_t, out_ptr, CUFFT_INVERSE);
      if (result != CUFFT_SUCCESS)
        throw CUFFTError (result, "CUDA::ConvolutionEngineSpectralCallbacks::perform_complex",
                          "cufftExecC2C(plan_bwd)");
      in_ptr  += ipol_stride;
      out_ptr += opol_stride;
    }
#else

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      result = cufftExecC2C (plan_fwd, in_ptr, buf_ptr, CUFFT_FORWARD);
      if (result != CUFFT_SUCCESS)
        throw CUFFTError (result, "CUDA::ConvolutionEngineSpectral::perform_complex",
                          "cufftExecC2C(plan_fwd)");

      in_ptr += ipol_stride;
      buf_ptr += bpol_stride;
    }

    // multiply by the dedispersion kernel for all polarisations
    if (dsp::Operation::verbose)
      cerr << "CUDA::ConvolutionEngineSpectral::perform_complex k_multiply_conv_spectral" << endl;
    k_multiply_conv_spectral<<<blocks, nthreads, 0, stream>>> (buf, d_kernels, npt_bwd, npol);

    buf_ptr = buf_t;
    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      // perform the inverse batched FFT (in-place)
      result = cufftExecC2C (plan_bwd, buf_ptr, buf_ptr, CUFFT_INVERSE);
      if (result != CUFFT_SUCCESS)
        throw CUFFTError (result, "CUDA::ConvolutionEngineSpectralCallbacks::perform_complex",
                          "cufftExecC2C(plan_bwd)");
      buf_ptr += bpol_stride;
    }

    // perfomr the ncopy kernel for both pols from PFT to FPT
    k_ncopy_conv_spectral_pft<<<blocks, nthreads, 0, stream>>> (out_ptr, opol_stride,
                                                                buf_t, npt_bwd,
                                                                nfilt_pos, nsamp_step, npol);
#endif

    // shift the input and output pointers forard by the overlap
    in_t  += nsamp_step;
    out_t += nsamp_step;
    if (output_zdm)
      out_zdm_t += nsamp_step;
  }

  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream( "CUDA::ConvolutionEngineSpectral::perform_complex", stream );
}
