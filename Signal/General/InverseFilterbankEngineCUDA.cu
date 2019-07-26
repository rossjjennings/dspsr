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

#include "CUFFTError.h"
#include "dsp/InverseFilterbankEngineCUDA.h"

void check_error (const char*);

/*!
 * Kernel for removing any overlap discard regions, optionally multiplying
 * by a response kernel in the process. Assumes input data is FPT order.
 * \method k_overlap_discard
 * \param t_in the input data array pointer. The shape of the array should be
 *    (nchan, ndat)
 * \param apodization the apodization kernel
 * \param t_out the output data array pointer
 * \param discard_pos the positive overlap discard region, in *complex samples*
 * \param discard_neg the negative overlap discard region, in *complex samples*
 * \param ndat the number of time samples in t_in
 * \param nchan the number of channels in t_in
 */
__global__ void k_overlap_save (
  float2* t_in,
  float2* t_out,
  int discard_pos,
  int discard_neg,
  int ipart_begin,
  int ipart_end,
  int npol,
  int nchan,
  int samples_per_part,
  int in_ndat,
  int out_ndat
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

  int step = samples_per_part - (discard_pos + discard_neg);
  int npart = ipart_end - ipart_begin;
  // make sure we're not trying to access data that are out of bounds
  if (idx > step || idy > nchan || idz > npol*npart) {
    return;
  }

  int in_offset;
  int out_offset;

  for (int ichan=idy; ichan<nchan; ichan+=total_size_y) {
    for (int ipol=idz%npol; ipol<npol; ipol+=npol_incr) {
      for (int ipart=idz/npol; ipart<npart; ipart+=npart_incr) {
        in_offset = ichan*npol*in_ndat + ipol*in_ndat + (ipart + ipart_begin)*samples_per_part;
        out_offset = ichan*npol*out_ndat + ipol*out_ndat + (ipart + ipart_begin)*step;

        for (int idat=idx; idat<step; idat += total_size_x) {
          t_out[out_offset + idat] = t_in[in_offset + idat + discard_pos];
        }
      }
    }
  }
}




/*!
 * Kernel for removing any overlap discard regions, optionally multiplying
 * by a response kernel in the process. Assumes input data is FPT order.
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
  int discard_pos,
  int discard_neg,
  int ipart_begin,
  int ipart_end,
  int npol,
  int nchan,
  int samples_per_part,
  int in_ndat,
  int out_ndat
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

  int step = samples_per_part - (discard_pos + discard_neg);
  int npart = ipart_end - ipart_begin;
  // make sure we're not trying to access data that are out of bounds
  if (idx > samples_per_part || idy > nchan || idz > npol*npart) {
    return;
  }

  int in_offset;
  int out_offset;

  for (int ichan=idy; ichan < nchan; ichan += total_size_y) {
    for (int ipol=idz%npol; ipol<npol; ipol+=npol_incr) {
      for (int ipart=idz/npol; ipart<npart; ipart+=npart_incr) {
        in_offset = ichan*npol*in_ndat + ipol*in_ndat + (ipart + ipart_begin)*step;
        out_offset = ichan*npol*out_ndat + ipol*out_ndat + (ipart + ipart_begin)*samples_per_part;
        for (int idat=idx; idat<samples_per_part; idat += total_size_x) {
          if (resp == nullptr) {
            t_out[out_offset + idat] = t_in[in_offset + idat];
          } else {
            t_out[out_offset + idat] = cuCmulf(resp[idat], t_in[in_offset + idat]);
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
  int ipart_begin,
  int ipart_end,
  int npol,
  int in_nchan,
  int in_ndat,
  int out_ndat,
  int in_samples_per_part,
  int out_samples_per_part,
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

  int npart = ipart_end - ipart_begin;

  int in_ndat_keep = in_samples_per_part - 2*os_discard;
  int in_ndat_keep_2 = in_ndat_keep / 2;

  // don't overstep the data
  if (idx > in_ndat_keep_2 || idy > in_nchan || idz > (npol*npart)) {
    return;
  }


  int in_offset;
  int out_offset;

  int in_idx_bot;
  int in_idx_top;

  int out_idx_bot;
  int out_idx_top;


  for (int ichan=idy; ichan < in_nchan; ichan += total_size_y) {
    for (int ipol=idz%npol; ipol < npol; ipol += npol_incr) {
      for (int ipart=idz/npol; ipart < npart; ipart += npart_incr) {
        in_offset = ichan*npol*in_ndat + ipol*in_ndat + (ipart + ipart_begin)*in_samples_per_part;
        out_offset = ipol*out_ndat + (ipart + ipart_begin)*out_samples_per_part;

        // in_offset = ipart*npol*in_ndat*in_nchan + ipol*in_ndat*in_nchan + ichan*in_ndat;
        // out_offset = ipart*npol*out_ndat + ipol*out_ndat;
        // std::cerr << "in_offset=" << in_offset << ", out_offset=" << out_offset << std::endl;

        for (int idat=idx; idat<in_ndat_keep_2; idat += total_size_x) {
          in_idx_top = idat;
          in_idx_bot = in_idx_top + (in_samples_per_part - in_ndat_keep_2);

          out_idx_bot = idat + in_ndat_keep*ichan;
          out_idx_top = out_idx_bot + in_ndat_keep_2;

          if (pfb_dc_chan) {
            if (ichan == 0) {
              out_idx_top = idat;
              out_idx_bot = idat + (out_samples_per_part - in_ndat_keep_2);
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
  forward_fft_plan_setup = false;
  backward_fft_plan_setup = false;
  response = nullptr;
  fft_window = nullptr;

  d_response = nullptr;
  d_fft_window = nullptr;

  pfb_dc_chan = 0;
  pfb_all_chan = 0;
  verbose = dsp::Observation::verbose;

  cufftHandle plans[] = {forward, backward};
  int nplans = sizeof(plans) / sizeof(plans[0]);
  cufftResult result;
  for (int i=0; i<nplans; i++) {
    result = cufftCreate (&plans[i]);
    if (verbose) {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::InverseFilterbankEngineCUDA: i=" << i << " result=" << result << std::endl;
    }
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
    if (verbose) {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::~InverseFilterbankEngineCUDA: i=" << i << " result=" << result << std::endl;
    }
    if (result == CUFFT_INVALID_PLAN) {
      if (verbose) {
        std::cerr << "CUDA::InverseFilterbankEngineCUDA::~InverseFilterbankEngineCUDA: plan[" << i << "] was invalid" << std::endl;
      }
      // throw CUFFTError (
      //   result,
      //   "CUDA::InverseFilterbankEngineCUDA::InverseFilterbankEngineCUDA",
      //   "cufftDestroy");
    }

  }
}

std::vector<cufftResult> CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan (
  unsigned _input_fft_length,
  unsigned _input_nchan,
  cufftType _type_forward
)
{
  // setup forward batched plan
  int rank = 1; // 1D transform
  int n[] = {_input_fft_length}; /* 1d transforms of length 10 */
  int howmany = _input_nchan;
  int idist = _input_fft_length;
  int odist = _input_fft_length;
  int istride = 1;
  int ostride = 1;
  int *inembed = n, *onembed = n;
  cufftResult result;
  std::vector<cufftResult> results;

  // cufftResult = cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
  //     int istride, int idist, int *onembed, int ostride,
  //     int odist, cufftType type, int batch)

  // result = cufftPlan1d (&forward, input_fft_length, type_forward, 1);
  result = cufftPlanMany(
    &forward, rank, n,
    inembed, istride, idist,
    onembed, ostride, odist,
    _type_forward, howmany);
  results.push_back(result);

  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan",
                      "cufftPlanMany(forward)");

  result = cufftSetStream (forward, stream);
  results.push_back(result);

  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan",
          "cufftSetStream(forward)");
  forward_fft_plan_setup = true;
  return results;
}

std::vector<cufftResult> CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan (
  unsigned _output_fft_length,
  unsigned _output_nchan
)
{
  // setup forward batched plan
  int rank = 1; // 1D transform
  int n[] = { _output_fft_length}; /* 1d transforms of length 10 */
  int howmany = _output_nchan;
  int idist = _output_fft_length;
  int odist = _output_fft_length;
  int istride = 1;
  int ostride = 1;
  int *inembed = n, *onembed = n;
  cufftResult result;
  std::vector<cufftResult> results;


  // result = cufftPlan1d (&backward, output_fft_length, CUFFT_C2C, 1);
  result = cufftPlanMany(
    &backward, rank, n,
    inembed, istride, idist,
    onembed, ostride, odist,
    CUFFT_C2C, howmany);

  results.push_back(result);

  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan",
                      "cufftPlan1d(backward)");

  result = cufftSetStream (backward, stream);
  results.push_back(result);
  if (result != CUFFT_SUCCESS)
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan",
                      "cufftSetStream(backward)");
  backward_fft_plan_setup = true;

  return results;
}


void CUDA::InverseFilterbankEngineCUDA::setup (
  const dsp::TimeSeries* input,
  dsp::TimeSeries* output,
  const Rational& os_factor,
  unsigned _input_fft_length,
  unsigned _output_fft_length,
  unsigned _input_discard_pos,
  unsigned _input_discard_neg,
  unsigned _output_discard_pos,
  unsigned _output_discard_neg,
  bool _pfb_dc_chan,
  bool _pfb_all_chan
)
{

  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup("
      << input << ", "
      << output << ", "
      << os_factor << ", "
      << _input_fft_length << ", "
      << _output_fft_length << ", "
      << _input_discard_pos << ", "
      << _input_discard_neg << ", "
      << _output_discard_pos << ", "
      << _output_discard_neg << ", "
      << _pfb_dc_chan << ", "
      << _pfb_all_chan << ")"
      << std::endl;
  }


  type_forward = CUFFT_C2C;
  n_per_sample = 1;

  if (input->get_state() == Signal::Nyquist) {
    type_forward = CUFFT_R2C;
    n_per_sample = 2;
  }

  pfb_dc_chan = _pfb_dc_chan;
  pfb_all_chan = _pfb_all_chan;

  input_npol = input->get_npol();
  input_nchan = input->get_nchan();
  output_nchan = output->get_nchan();

  input_fft_length = _input_fft_length;
  output_fft_length = _output_fft_length;

  input_discard_pos = _input_discard_pos;
  input_discard_neg = _input_discard_neg;
  output_discard_pos = _output_discard_pos;
  output_discard_neg = _output_discard_neg;

  input_discard_total = n_per_sample*(input_discard_neg + input_discard_pos);
  input_sample_step = input_fft_length - input_discard_total;

  output_discard_total = n_per_sample*(output_discard_neg + output_discard_pos);
  output_sample_step = output_fft_length - output_discard_total;

  input_os_keep = os_factor.normalize(input_fft_length);
  input_os_discard = input_fft_length - input_os_keep;

  setup_forward_fft_plan(
    input_fft_length, input_nchan, type_forward
  );

  setup_backward_fft_plan(
    output_fft_length, output_nchan
  );

  // need device memory for response
  if (response) {
  
	}
  // need device memory for apodization
  if (fft_window) {

  }

  // if (stream)
  //   cudaMemcpyAsync(d_kernel, kernel, mem_size, cudaMemcpyHostToDevice, stream);
  // else
  //   cudaMemcpy (d_kernel, kernel, mem_size, cudaMemcpyHostToDevice);

  // now setup scratch space.
  float2* d_scratch;

  unsigned d_input_overlap_discard_size = input_npol*input_nchan*input_fft_length;
  unsigned d_stitching_size = input_npol*output_nchan*output_fft_length;

  unsigned scratch_needed = d_input_overlap_discard_size + d_stitching_size;

  cudaMalloc((void**)&d_scratch, sizeof(float2)*scratch_needed);

  d_input_overlap_discard = d_scratch;
  d_stitching = d_scratch + d_input_overlap_discard_size;

}



void CUDA::InverseFilterbankEngineCUDA::setup (dsp::InverseFilterbank* filterbank)
{
  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup("
      << filterbank << ")"
      << std::endl;
  }
  const dsp::TimeSeries* input = filterbank->get_input();
  dsp::TimeSeries* output = filterbank->get_output();

  setup (
    input,
    output,
    filterbank->get_oversampling_factor(),
    filterbank->get_input_fft_length(),
    filterbank->get_output_fft_length(),
    filterbank->get_input_discard_pos(),
    filterbank->get_input_discard_neg(),
    filterbank->get_output_discard_pos(),
    filterbank->get_output_discard_neg(),
    filterbank->get_pfb_dc_chan(),
    filterbank->get_pfb_all_chan()
  );
}


void CUDA::InverseFilterbankEngineCUDA::set_scratch (float* )
{ }


void CUDA::InverseFilterbankEngineCUDA::perform (
  const dsp::TimeSeries* in,
  dsp::TimeSeries* out,
  uint64_t npart
  // uint64_t in_step,
  // uint64_t out_step
)
{
  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform("
      << in << ", "
      << out << ", "
      << npart << ")"
      << std::endl;
  }
  // in_step is input_sample_step
  // out_step is output_sample_step
	

	// get_datptr (unsigned ichan, unsigned ipol) 
	// we can't use TimeSeries::get_ndat because there might be some buffer space that ndat 
	// doesn't account for 
  unsigned input_stride = (in->get_datptr (1, 0) - in->get_datptr (0, 0)) / n_per_sample;
  unsigned output_stride = (out->get_datptr (1, 0) - out->get_datptr (0, 0) ) / 2;

  const float* in_ptr = in->get_datptr(0, 0);
  float* out_ptr = out->get_datptr(0, 0);

  dim3 grid (1, input_nchan, input_npol*npart);
  dim3 threads (1024, 1, 1);


  for (unsigned ipart=0; ipart<npart; ipart++)
  {
	
		if (type_forward == CUFFT_R2C) 
		{
			throw "Currently incompatible with R2C"; 
      // k_overlap_discard<<<grid, threads, 0, stream>>> (
      //   (cufftReal*) in_ptr, d_fft_window,
      //   (cufftReal*) d_input_overlap_discard,
      //   input_discard_pos, input_discard_neg,
      //   ipart, ipart + 1, input_npol, input_nchan,
      //   input_fft_length, input_stride, 0
      // );
      // cufftExecR2C(
      //   forward,
      //   (cufftReal*) d_input_overlap_discard,
      //   (cufftComplex*) d_input_overlap_discard
      // );
		}
		else
		{
      k_overlap_discard<<<grid, threads, 0, stream>>> (
        (cufftComplex*) in_ptr, d_fft_window,
        (cufftComplex*) d_input_overlap_discard,
        input_discard_pos, input_discard_neg,
        ipart, ipart + 1, input_npol, input_nchan,
        input_fft_length, input_stride, 0
      );
     	cufftExecC2C(
     	  forward,
     	 	(cufftComplex*) d_input_overlap_discard,
     	  (cufftComplex*) d_input_overlap_discard,
     	  CUFFT_FORWARD
     	);
		}

    k_response_stitch<<<grid, threads, 0, stream>>>(
      (float2*) d_input_overlap_discard, d_response, (float2*) d_stitching,
      input_os_discard/2, ipart, ipart + 1, input_npol, input_nchan,
      npart * input_fft_length, npart * output_nchan * output_fft_length,
      input_fft_length, output_nchan * output_fft_length,
      pfb_dc_chan, pfb_all_chan
    );

    cufftExecC2C (
      backward,
      (cufftComplex*) d_input_overlap_discard,
      (cufftComplex*) d_input_overlap_discard,
      CUFFT_INVERSE
    );

    k_overlap_save<<<grid, threads, 0, stream>>>(
      (float2* ) d_input_overlap_discard,
      (float2*) out_ptr,
      output_discard_pos,
      output_discard_neg,
      ipart, ipart+1, input_npol, output_nchan,
      output_fft_length,
      npart * output_nchan * output_fft_length,
      output_stride
    );
  }
}

void CUDA::InverseFilterbankEngineCUDA::finish ()
{
  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::finish" << std::endl;
  }

  cudaError_t error = cudaFree (d_scratch);
  if (error != cudaSuccess)
     throw Error (FailedCall, "CUDA::InverseFilterbankEngineCUDA::finish",
                 "cudaFree(%xu): %s", &d_scratch,
                 cudaGetErrorString (error));

}


//! This method is static
void CUDA::InverseFilterbankEngineCUDA::apply_k_response_stitch (
  std::vector< std::complex<float> >& in,
  std::vector< std::complex<float> >& response,
  std::vector< std::complex<float> >& out,
  Rational os_factor,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned ndat,
  bool pfb_dc_chan,
  bool pfb_all_chan)
{
  float2* in_device;
  float2* resp_device;
  float2* out_device;

  unsigned in_ndat = ndat;
  unsigned os_keep = os_factor.normalize(in_ndat);
  unsigned os_discard = (in_ndat - os_keep)/2;
  unsigned out_ndat = nchan * os_keep;
  unsigned out_size = npart * npol * out_ndat;
  unsigned in_size = npart * npol * nchan * ndat;

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
    in_device, resp_device, out_device, os_discard, 0, npart,
    npol, nchan, npart*in_ndat, npart*out_ndat, ndat, out_ndat,
    pfb_dc_chan, pfb_all_chan);

  check_error( "CUDA::InverseFilterbankEngineCUDA::apply_k_response_stitch" );

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
  unsigned discard,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned ndat
)
{
  float2* in_device;
  float2* apod_device;
  float2* out_device;

  size_t sz = sizeof(float2);

  unsigned total_discard = 2*discard;
  unsigned step = ndat - total_discard;

  unsigned in_ndat = npart * step + total_discard;
  unsigned out_ndat = npart * ndat;

  unsigned in_size = npol * nchan * in_ndat;
  unsigned out_size = npol * nchan * out_ndat;

  cudaMalloc((void **) &in_device, in_size*sz);
  cudaMalloc((void **) &apod_device, ndat*sz);
  cudaMalloc((void **) &out_device, out_size*sz);

  cudaMemcpy(
    in_device, (float2*) in.data(), in_size*sz, cudaMemcpyHostToDevice);
  cudaMemcpy(
    apod_device, (float2*) apodization.data(), ndat*sz, cudaMemcpyHostToDevice);

  dim3 grid (1, nchan, npol*npart);
  dim3 threads (1024, 1, 1);
  grid.x = (ndat / threads.x) + 1;


  k_overlap_discard<<<grid, threads>>>(
    in_device, apod_device, out_device, discard, discard, 0, npart, npol, nchan, ndat, in_ndat, out_ndat);
  check_error( "CUDA::InverseFilterbankEngineCUDA::apply_k_apodization_overlap" );

  cudaMemcpy((float2*) out.data(), out_device, out_size*sz, cudaMemcpyDeviceToHost);

  cudaFree(in_device);
  cudaFree(apod_device);
  cudaFree(out_device);

}


void CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard (
  std::vector< std::complex<float> >& in,
  std::vector< std::complex<float> >& out,
  unsigned discard,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned ndat
)
{
  float2* in_device;
  float2* out_device;

  size_t sz = sizeof(float2);

  unsigned total_discard = 2*discard;
  unsigned step = ndat - total_discard;

  unsigned in_ndat = npart * step + total_discard;
  unsigned out_ndat = npart * ndat;

  unsigned in_size = npol * nchan * in_ndat;
  unsigned out_size = npol * nchan * out_ndat;

  cudaMalloc((void **) &in_device, in_size*sz);
  cudaMalloc((void **) &out_device, out_size*sz);

  cudaMemcpy(
    in_device, (float2*) in.data(), in_size*sz, cudaMemcpyHostToDevice);

  dim3 grid (1, nchan, npol*npart);
  dim3 threads (1024, 1, 1);

  grid.x = (ndat / threads.x) + 1;

  // std::cerr << grid.x << " " << grid.y << " " << grid.z << std::endl;
  // std::cerr << threads.x << " " << threads.y << " " << threads.z << std::endl;

  k_overlap_discard<<<grid, threads>>>(
    in_device, nullptr, out_device, discard, discard, 0, npart, npol, nchan, ndat, in_ndat, out_ndat);

  check_error( "CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard" );

  cudaMemcpy((float2*) out.data(), out_device, out_size*sz, cudaMemcpyDeviceToHost);

  cudaFree(in_device);
  cudaFree(out_device);
}


void CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_save (
  std::vector< std::complex<float> >& in,
  std::vector< std::complex<float> >& out,
  unsigned discard,
  unsigned npart,
  unsigned npol,
  unsigned nchan,
  unsigned ndat
)
{
  float2* in_device;
  float2* out_device;

  size_t sz = sizeof(float2);

  unsigned total_discard = 2*discard;
  unsigned step = ndat - total_discard;

  unsigned in_ndat = npart * ndat;
  unsigned out_ndat = npart * step;

  unsigned in_size = npol * nchan * in_ndat;
  unsigned out_size = npol * nchan * out_ndat;

  cudaMalloc((void **) &in_device, in_size*sz);
  cudaMalloc((void **) &out_device, out_size*sz);

  cudaMemcpy(
    in_device, (float2*) in.data(), in_size*sz, cudaMemcpyHostToDevice);

  dim3 grid (1, nchan, npol*npart);
  dim3 threads (1024, 1, 1);

  grid.x = (ndat / threads.x) + 1;

  // std::cerr << grid.x << " " << grid.y << " " << grid.z << std::endl;
  // std::cerr << threads.x << " " << threads.y << " " << threads.z << std::endl;

  k_overlap_save<<<grid, threads>>>(
    in_device, out_device, discard, discard, 
    0, npart, npol, nchan, ndat, in_ndat, out_ndat);

  check_error( "CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard" );

  cudaMemcpy((float2*) out.data(), out_device, out_size*sz, cudaMemcpyDeviceToHost);

  cudaFree(in_device);
  cudaFree(out_device);
}

void CUDA::InverseFilterbankEngineCUDA::apply_cufft_backward (
  std::vector< std::complex<float> >& in,
  std::vector< std::complex<float> >& out
)
{
  if (! backward_fft_plan_setup) {
    throw "CUDA::InverseFilterbankEngineCUDA::apply_cufft_backward: Backward FFT plan not setup";
  }

  cufftComplex* in_cufft;
  cufftComplex* out_cufft;

  size_t sz = sizeof(cufftComplex);

  cudaMalloc((void **) &in_cufft, in.size()*sz);
  cudaMalloc((void **) &out_cufft, out.size()*sz);

  cudaMemcpy(
    in_cufft, (cufftComplex*) in.data(), in.size()*sz, cudaMemcpyHostToDevice);

  cufftExecC2C(backward, in_cufft, out_cufft, CUFFT_INVERSE);
  cudaDeviceSynchronize();
  check_error( "CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_discard" );

  cudaMemcpy(
    (cufftComplex*) out.data(), out_cufft, out.size()*sz, cudaMemcpyDeviceToHost);

  cudaFree(in_cufft);
  cudaFree(out_cufft);

}
