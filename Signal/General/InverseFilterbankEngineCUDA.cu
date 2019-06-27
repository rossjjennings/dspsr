//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten, Andrew Jameson and Dean Shaff
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbankEngineCUDA.h"

#include <cuda_runtime.h>

/*!
 * Kernel for multiplying a time domain array by an apodization kernel,
 * removing any overlap discard regions in the process.
 * \method k_apodization_overlap
 * \param t_in the input data array pointer
 * \param apodization the apodization kernel
 * \param t_out the output data array pointer
 * \param discard the overlap discard region, in *complex samples*
 */
__global__ void k_apodization_overlap (
  float2* t_in,
  float2* apodization,
  float2* t_out,
  int discard
)
{ }


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
 * \param oversampled_discard tthe number of *complex samples* to discard
 *    from either side of the input spectra.
 */
__global__ void k_response_stitch (
  float2* f_in,
  float2* response,
  float2* f_out,
  int oversampled_discard
)
{ }


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
{ }


CUDA::InverseFilterbankEngineCUDA::InverseFilterbankEngineCUDA (cudaStream_t stream)
{ }

CUDA::InverseFilterbankEngineCUDA::~InverseFilterbankEngineCUDA ()
{ }

void CUDA::InverseFilterbankEngineCUDA::setup (dsp::InverseFilterbank* filterbank)
{ }

double CUDA::InverseFilterbankEngineCUDA::setup_fft_plans (dsp::InverseFilterbank* filterbank)
{
  return 0.0;
}

void CUDA::InverseFilterbankEngineCUDA::set_scratch (float* )
{ }

void CUDA::InverseFilterbankEngineCUDA::perform (const dsp::TimeSeries* in, dsp::TimeSeries* out,
              uint64_t npart, uint64_t in_step, uint64_t out_step)
{ }

void CUDA::InverseFilterbankEngineCUDA::finish ()
{ }
