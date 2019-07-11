//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2019 by Willem van Straten, Andrew Jameson and Dean Shaff
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <cstdio>
#include <vector>
#include <complex>

#include <cuda_runtime.h>

#include "CUFFTError.h"
#include "dsp/InverseFilterbankEngineCUDA.h"


/*!
 * Kernel for removing any overlap discard regions, optionally multiplying
 * by a response kernel in the process.
 * \method k_overlap_discard
 * \param t_in the input data array pointer. The shape of the array should be
 *    (nchan, ndat)
 * \param apodization the apodization kernel
 * \param t_out the output data array pointer
 * \param discard the overlap discard region, in *complex samples*
 * \param ndat the number of time samples in t_in
 * \param nchan the number of channels in t_in
 */
__global__ void k_overlap_discard (
  float2* t_in,
  float2* resp,
  float2* t_out,
  int discard,
  int npart,
  int npol,
  int nchan,
  int ndat
)
{
  int total_size_x = blockDim.x * gridDim.x; // for ndat
  int total_size_y = blockDim.y * gridDim.y; // for nchan
  int total_size_z = blockDim.z * gridDim.z; // for npart and npol
  int npol_incr = total_size_z <= npol ? 1: npol;
  int npart_incr = total_size_z/npol == 0 ? 1: total_size_z/npol;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  int idz = blockIdx.z*blockDim.z + threadIdx.z;

  // if (npol == 2){
  //   printf("total_size_z=%d, npol_incr=%d, idz=%d, idz%%npol=%d, idz/npol=%d\n",
  //     total_size_z, npol_incr, idz, idz%npol, idz/npol
  //   );
  // }

  int out_ndat = ndat - 2*discard;

  // make sure we're not trying to access channels that don't exist
  if (idx > out_ndat || idy > nchan || idz > npol*npart) {
    return;
  }

  int in_offset;
  int out_offset;

  for (int ipart=idz/npol; ipart<npart; ipart+=npart_incr) {
    for (int ipol=idz%npol; ipol<npol; ipol+=npol_incr) {
      for (int ichan=idy; ichan < nchan; ichan += total_size_y) {
        for (int idat=idx; idat < out_ndat; idat += total_size_x) {
          in_offset = ipart*npol*nchan*ndat + ipol*nchan*ndat + ichan*ndat;
          out_offset = ipart*npol*nchan*out_ndat + ipol*nchan*out_ndat + ichan*out_ndat;
          if (resp == nullptr) {
            t_out[out_offset + idat] = t_in[in_offset + idat + discard];
          } else {
            t_out[out_offset + idat] = cuCmulf(resp[ichan*out_ndat + idat], t_in[in_offset + idat + discard]);
          }
        }
      }
    }
  }
}

/*!
 * fft shift an index. Returns -1 if ``idx`` is greater than ``ndat``
 * \method d_fft_shift_idx
 * \param idx the index to shift
 * \ndat the number of points about which to shift
 * \return circularly shifted index.
 */
__device__ int d_fft_shift_idx (int idx, int ndat)
{
  int ndat_2 = ndat / 2;
  if (idx >= ndat) {
    return -1;
  }
  if (idx >= ndat_2) {
    return idx - ndat_2;
  } else {
    return idx + ndat_2;
  }
}


/*!
 * Kernel for stitching together the result of forward FFTs, and multiplying
 * Response or ResponseProduct's internal buffer by stitched result.
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
 * \param os_discard the number of *complex samples* to discard
 *    from either side of the input spectra.
 * \param npol the number of polarisations
 * \param in_nchan the number of channels in the input data. The first dimension
 *    of the input data is in_nchan*npol.
 * \param in_ndat the second dimension of the input data.
 * \param out_ndat the second dimension of the output data.
 * \param pfb_dc_chan whether or not the DC channel of the PFB channeliser is
 *    present.
 * \param pfb_all_chan whether or not all the channels from the PFB channeliser
 *    are present.
 */
__global__ void k_response_stitch (
  float2* f_in,
  float2* response,
  float2* f_out,
  int os_discard,
  int npart,
  int npol,
  int in_nchan,
  int in_ndat,
  int out_ndat,
  bool pfb_dc_chan,
  bool pfb_all_chan
)
{
  int total_size_x = blockDim.x * gridDim.x; // for idat
  int total_size_y = blockDim.y * gridDim.y; // for ichan
  int total_size_z = blockDim.z * gridDim.z; // for ipol and ipart
  int npol_incr = total_size_z <= npol ? 1: npol;
  int npart_incr = total_size_z/npol == 0 ? 1: total_size_z/npol;


  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  int idz = blockIdx.z*blockDim.z + threadIdx.z;

  // don't overstep the data
  if (idx > in_ndat - os_discard || idy > in_nchan || idz > (npol*npart)) {
    return;
  }

  int in_ndat_keep = in_ndat - 2*os_discard;
  int in_ndat_keep_2 = in_ndat_keep / 2;

  int in_offset;
  int out_offset;

  int in_idx_bot;
  int in_idx_top;

  int out_idx_bot;
  int out_idx_top;


  for (int ipart=idz/npol; ipart < npart; ipart += npart_incr) {
    for (int ipol=idz%npol; ipol < npol; ipol += npol_incr) {
      for (int ichan=idy; ichan < in_nchan; ichan += total_size_y) {
        in_offset = ipart*npol*in_ndat*in_nchan + ipol*in_ndat*in_nchan + ichan*in_ndat;
        out_offset = ipart*npol*out_ndat + ipol*out_ndat;
        // std::cerr << "in_offset=" << in_offset << ", out_offset=" << out_offset << std::endl;

        for (int idat=idx; idat<in_ndat_keep_2; idat += total_size_x) {
          in_idx_top = idat;
          in_idx_bot = in_idx_top + (in_ndat - in_ndat_keep_2);

          out_idx_bot = idat + in_ndat_keep*ichan;
          out_idx_top = out_idx_bot + in_ndat_keep_2;

          if (pfb_dc_chan) {
            if (ichan == 0) {
              out_idx_top = idat;
              out_idx_bot = idat + (out_ndat - in_ndat_keep_2);
            } else {
              out_idx_bot = idat + in_ndat_keep*(ichan-1) + in_ndat_keep_2;
              out_idx_top = out_idx_bot + in_ndat_keep_2;
            }
          }

          // std::cerr << in_offset + in_idx_bot << ", " << in_offset + in_idx_top << std::endl;
          // std::cerr << out_offset + out_idx_bot << ", " << out_offset + out_idx_top << std::endl;
          //
          // if (in_offset + in_idx_bot > in_size ||
          //     out_offset + out_idx_top > out_size ||
          //     in_offset + in_idx_top > in_size ||
          //     out_offset + out_idx_bot > out_size) {
          //   std::cerr << "watch out!" << std::endl;
          // }
          // std::cerr << "in=[" << in_idx_bot << "," << in_idx_top << "] out=["
          //   << out_idx_bot << "," << out_idx_top << "]" << std::endl;

          f_out[out_offset + out_idx_bot] = cuCmulf(response[out_idx_bot], f_in[in_offset + in_idx_bot]);
          f_out[out_offset + out_idx_top] = cuCmulf(response[out_idx_top], f_in[in_offset + in_idx_top]);

          if (! pfb_all_chan && pfb_dc_chan && ichan == 0) {
            f_out[out_offset + out_idx_bot].x = 0.0;
            f_out[out_offset + out_idx_bot].y = 0.0;
          }
        }
      }
    }
  }
  // for (int ipol=idz; ipol < npol; ipol += total_size_z) {
  //   for (int ichan=idy; ichan < in_nchan; ichan += total_size_y) {
  //     for (int idat=idx; idat < (in_ndat - os_discard); idat += total_size_x) {
  //       if (idat < os_discard) {
  //         continue;
  //       }
  //       in_offset = ipol*in_nchan*in_ndat + ichan*in_ndat;
  //       out_offset = ipol*out_ndat;
  //
  //       in_idx_top = in_offset + (idat - os_discard);
  //       in_idx_bot = in_idx_top + (in_ndat - in_ndat_keep_2);
  //
  //       out_idx_bot = ichan*in_ndat_keep + (idat - os_discard);
  //       out_idx_top = out_idx_bot + in_ndat_keep_2;
  //
  //       if (pfb_dc_chan) {
  //         if (ichan == 0) {
  //           out_idx_bot = idat - os_discard;
  //           out_idx_top = out_idx_bot + out_ndat - in_ndat_keep_2;
  //         } else {
  //           out_idx_bot += in_ndat_keep_2;
  //           out_idx_top += in_ndat_keep_2;
  //         }
  //       }
  //       f_out[out_idx_bot + out_offset] = cuCmulf(response[out_idx_bot], f_in[in_idx_bot]);
  //       f_out[out_idx_top + out_offset] = cuCmulf(response[out_idx_top], f_in[in_idx_top]);
  //
  //       if (pfb_dc_chan && ! pfb_all_chan) {
  //         f_out[out_idx_top + out_offset].x = 0.0;
  //         f_out[out_idx_top + out_offset].y = 0.0;
  //       }
  //     }
  //   }
  // }
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
void CUDA::InverseFilterbankEngineCUDA::apply_k_response_stitch (
  std::vector< std::complex<float> >& in,
  std::vector< std::complex<float> >& response,
  std::vector< std::complex<float> >& out,
  Rational os_factor,
  int npart,
  int npol,
  int nchan,
  int ndat,
  bool pfb_dc_chan,
  bool pfb_all_chan)
{
  float2* in_device;
  float2* resp_device;
  float2* out_device;

  int in_ndat = ndat;
  int os_keep = os_factor.normalize(in_ndat);
  int os_discard = (in_ndat - os_keep)/2;
  int out_ndat = nchan * os_keep;
  int out_size = npart * npol * out_ndat;
  int in_size = npart * npol * nchan * ndat;

  if (out.size() != out_size) {
    out.resize(out_size);
  }

  size_t sz = sizeof(float2);

  cudaMalloc((void **) &in_device, in_size*sz);
  cudaMalloc((void **) &resp_device, out_ndat*sz);
  cudaMalloc((void **) &out_device, out_size*sz);

  cudaMemcpy(
    in_device, (float2*) in.data(), in_size*sz, cudaMemcpyHostToDevice);
  cudaMemcpy(
    resp_device, (float2*) response.data(), out_ndat*sz, cudaMemcpyHostToDevice);

  // 10 is sort of arbitrary here.
  dim3 grid (1, nchan, npart*npol);
  dim3 threads (in_ndat, 1, 1);

  k_response_stitch<<<grid, threads>>>(
    in_device, resp_device, out_device, os_discard, npart,
    npol, nchan, in_ndat, out_ndat, pfb_dc_chan, pfb_all_chan);

  if (dsp::Operation::verbose) {
    // check_error ("CUDA::InverseFilterbankEngineCUDA::apply_k_response_stitch");
  }

  cudaMemcpy((float2*) out.data(), out_device, out_size*sz, cudaMemcpyDeviceToHost);

  cudaFree(in_device);
  cudaFree(resp_device);
  cudaFree(out_device);
}

//! This method is static
void CUDA::InverseFilterbankEngineCUDA::apply_k_apodization_overlap (
  std::vector< std::complex<float> >& in,
  std::vector< std::complex<float> >& apodization,
  std::vector< std::complex<float> >& out,
  int discard,
  int npart,
  int npol,
  int nchan,
  int ndat
)
{
  float2* in_device;
  float2* apod_device;
  float2* out_device;

  size_t sz = sizeof(float2);

  int out_ndat = ndat - 2*discard;
  int in_size = npart * npol * nchan * ndat;
  int out_size = npart * npol * nchan * out_ndat;
  int apod_size = nchan * out_ndat;

  cudaMalloc((void **) &in_device, in_size*sz);
  cudaMalloc((void **) &apod_device, apod_size*sz);
  cudaMalloc((void **) &out_device, out_size*sz);

  cudaMemcpy(
    in_device, (float2*) in.data(), in_size*sz, cudaMemcpyHostToDevice);
  cudaMemcpy(
    apod_device, (float2*) apodization.data(), apod_size*sz, cudaMemcpyHostToDevice);

  // 10 is sort of arbitrary here.
  dim3 grid (1, nchan, npol*npart);
  dim3 threads (1024, 1, 1);
  grid.x = (ndat / threads.x) + 1;


  k_overlap_discard<<<grid, threads>>>(
    in_device, apod_device, out_device, discard, npart, npol, nchan, ndat);

  cudaMemcpy((float2*) out.data(), out_device, out_size*sz, cudaMemcpyDeviceToHost);

  cudaFree(in_device);
  cudaFree(apod_device);
  cudaFree(out_device);
}


void CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard (
  std::vector< std::complex<float> >& in,
  std::vector< std::complex<float> >& out,
  int discard,
  int npart,
  int npol,
  int nchan,
  int ndat
)
{
  float2* in_device;
  float2* out_device;

  size_t sz = sizeof(float2);

  int out_ndat = ndat - 2*discard;

  int in_size = npart * npol * nchan * ndat;
  int out_size = npart * npol * nchan * out_ndat;

  cudaMalloc((void **) &in_device, in_size*sz);
  cudaMalloc((void **) &out_device, out_size*sz);

  cudaMemcpy(
    in_device, (float2*) in.data(), in_size*sz, cudaMemcpyHostToDevice);

  dim3 grid (1, nchan, npol*npart);
  dim3 threads (1024, 1, 1);

  grid.x = (ndat / threads.x) + 1;


  k_overlap_discard<<<grid, threads>>>(
    in_device, nullptr, out_device, discard, npart, npol, nchan, ndat);

  cudaMemcpy((float2*) out.data(), out_device, out_size*sz, cudaMemcpyDeviceToHost);

  cudaFree(in_device);
  cudaFree(out_device);
}
