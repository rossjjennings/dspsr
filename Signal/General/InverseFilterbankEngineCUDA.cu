//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten, Andrew Jameson and Dean Shaff
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "CUFFTError.h"
#include "dsp/InverseFilterbankEngineCUDA.h"


/*!
 * Kernel for multiplying a time domain array by an apodization kernel,
 * removing any overlap discard regions in the process.
 * \method k_apodization_overlap
 * \param t_in the input data array pointer. The shape of the array should be
 *    (nchan, ndat)
 * \param apodization the apodization kernel
 * \param t_out the output data array pointer
 * \param discard the overlap discard region, in *complex samples*
 * \param ndat the number of time samples in t_in
 * \param nchan the number of channels in t_in
 */
__global__ void k_apodization_overlap (
  float2* t_in,
  float2* apodization,
  float2* t_out,
  int discard,
  int ndat,
  int nchan
)
{
  int total_size_x = blockDim.x * gridDim.x;
  int total_size_y = blockDim.y * gridDim.y;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int apod_ndat = ndat - 2*discard;
  // printf("ndat=%d, nchan=%d, idx=%d, total_size_x=%d, total_size_y=%d\n", ndat, nchan, idx, total_size_x, total_size_y);

  // make sure we're not trying to access channels that don't exist
  if (blockIdx.y > nchan) {
    return;
  }
  // ignore top of overlap region
  if (idx > (ndat - discard)) {
    return;
  }

  for (int ichan=blockIdx.y; ichan < nchan; ichan += total_size_y) {
    for (int idat=idx; idat < (ndat - discard); idat += total_size_x) {
      // printf("ichan=%d, nchan=%d, idat=%d, t_in=(%f, %f), apodization=(%f, %f)\n",
      //   ichan, nchan, idat, t_in[ichan*ndat + idat].x, t_in[ichan*ndat + idat].y,
      //   apodization[idat-discard].x, apodization[idat-discard].y);
      // ignore bottom of overlap region
      if (idat < discard) {
        continue;
      }
      t_out[ichan*apod_ndat + (idat - discard)] = cuCmulf(apodization[idat - discard], t_in[ichan*ndat + idat]);
    }
  }
}


/*!
 * Kernel for multiplying a Response or ResponseProduct's internal
 * buffer by result of forward FFTs. After multiplying, this stitches
 * output spectra together.
 * \method k_response_stitch
 * \param f_in the frequency domain input data pointer.
 *    Dimensions are (npol*input_nchan, input_ndat).
 *    Here, input_ndat is equal to the size of the forward FFT.
 * \param response the Response or ResponseProduct's buffer
 * \param f_out the frequency domain output data pointer.
 *    Dimensions are (npol*1, output_ndat).
 *    Here, output_ndat is equal to the size of the backward FFT,
 *    which is in turn equal to input_nchan * input_ndat normalized by
 *    the oversampling factor.
 * \param oversampled_discard the number of *complex samples* to discard
 *    from either side of the input spectra.
 */
__global__ void k_response_stitch (
  float2* f_in,
  float2* response,
  float2* f_out,
  int oversampled_discard
)
{

}


/*!
 * Kernel for discarding overlap regions on output time domain data.
 * \method k_overlap
 * \param t_in input time domain pointer
 * \param t_out output time domain pointer
 * \param discard discard region on output data.
 */
__global__ void k_overlap (
  float2* t_in,
  float2* t_out,
  int discard
)
{

}

CUDA::InverseFilterbankEngineCUDA::InverseFilterbankEngineCUDA (cudaStream_t _stream)
{
  stream = _stream;

  input_fft_length = 0;
  fft_plans_setup = false;
  response = nullptr;
  fft_window = nullptr;

  pfb_dc_chan = 0;
  pfb_all_chan = 0;
  verbose = dsp::Observation::verbose;

  cufftHandle plans[] = {forward, backward};
  int nplans = sizeof(plans) / sizeof(plans[0]);
  cufftResult result;
  for (int i=0; i<nplans; i++) {
    result = cufftCreate (&plans[i]);
    if (result != CUFFT_SUCCESS) {
      throw CUFFTError (
        result,
        "CUDA::InverseFilterbankEngineCUDA::InverseFilterbankEngineCUDA",
        "cufftCreate");
    }
  }
}

CUDA::InverseFilterbankEngineCUDA::~InverseFilterbankEngineCUDA ()
{
  cufftHandle plans[] = {forward, backward};
  int nplans = sizeof(plans) / sizeof(plans[0]);

  cufftResult result;
  for (int i=0; i<nplans; i++) {
    result = cufftDestroy (plans[i]);
    if (result != CUFFT_SUCCESS) {
      throw CUFFTError (
        result,
        "CUDA::InverseFilterbankEngineCUDA::InverseFilterbankEngineCUDA",
        "cufftDestroy");
    }
  }
}

void CUDA::InverseFilterbankEngineCUDA::setup (dsp::InverseFilterbank* filterbank)
{
  if (filterbank->get_input()->get_state() == Signal::Nyquist) {
    type_forward = CUFFT_R2C;
  } else {
    type_forward = CUFFT_C2C;
  }
}

double CUDA::InverseFilterbankEngineCUDA::setup_fft_plans (dsp::InverseFilterbank* filterbank)
{
  // taken from ConvolutionCUDA engine.
  if (dsp::Operation::verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup_fft_plans"
      << " input_fft_length=" << input_fft_length
      << " output_fft_length=" << output_fft_length << std::endl;
  }

  // setup forward plan
  cufftResult result = cufftPlan1d (&forward, input_fft_length, type_forward, 1);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_fft_plans",
                      "cufftPlan1d(forward)");

  result = cufftSetStream (forward, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_fft_plans",
          "cufftSetStream(forward)");

  // setup backward plan
  result = cufftPlan1d (&backward, output_fft_length, CUFFT_C2C, 1);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_fft_plans",
                      "cufftPlan1d(backward)");

  result = cufftSetStream (backward, stream);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_fft_plans",
                      "cufftSetStream(backward)");

  // size_t buffer_size = output_fft_length * sizeof (cufftComplex);
  // cudaError_t error = cudaMalloc ((void **) &buf, buffer_size);
  // if (error != cudaSuccess)
  //   throw Error (FailedCall, "CUDA::InverseFilterbankEngineCUDA::setup_fft_plans",
  //                "cudaMalloc(%x, %u): %s", &buf, buffer_size,
  //                cudaGetErrorString (error));

  // Compute FFT scale factors
  scalefac = 1.0;
  if (FTransform::get_norm() == FTransform::unnormalized) {
    scalefac = pow(double(output_fft_length), 2);
    scalefac *= pow(filterbank->get_oversampling_factor().doubleValue(), 2);
  }
  fft_plans_setup = true;
  if (verbose) {
    std::cerr << "dsp::InverseFilterbankEngineCPU::setup_fft_plans: scalefac=" << scalefac << std::endl;
  }

  return scalefac;
}

void CUDA::InverseFilterbankEngineCUDA::set_scratch (float* )
{ }

void CUDA::InverseFilterbankEngineCUDA::perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
              uint64_t npart, uint64_t in_step, uint64_t out_step)
{ }

void CUDA::InverseFilterbankEngineCUDA::finish ()
{ }

//! This method is static
void CUDA::InverseFilterbankEngineCUDA::apply_k_apodization_overlap (
  std::complex<float>* in,
  std::complex<float>* apodization,
  std::complex<float>* out,
  int discard,
  int ndat,
  int nchan)
{
  float2* in_device;
  float2* apod_device;
  float2* out_device;

  size_t sz = sizeof(float2);
  int ndat_apod = (ndat - 2*discard);

  cudaMalloc((void **) &in_device, ndat*nchan*sz);
  cudaMalloc((void **) &apod_device, ndat_apod*sz);
  cudaMalloc((void **) &out_device, ndat_apod*nchan*sz);

  cudaMemcpy(
    in_device, (float2*) in, ndat*nchan*sz, cudaMemcpyHostToDevice);
  cudaMemcpy(
    apod_device, (float2*) apodization, ndat_apod*sz, cudaMemcpyHostToDevice);

  // 10 is sort of arbitrary here.
  dim3 grid (10, nchan, 1);
  dim3 threads (64, 1, 1);


  k_apodization_overlap<<<grid, threads>>>(
    in_device, apod_device, out_device, discard, ndat, nchan);

  cudaMemcpy((float2*) out, out_device, ndat_apod*nchan*sz, cudaMemcpyDeviceToHost);

  cudaFree(in_device);
  cudaFree(apod_device);
  cudaFree(out_device);
}
