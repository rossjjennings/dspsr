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

void check_error_stream (const char*, cudaStream_t);


__global__ void k_print_array (
  float2* in,
  int nchan,
  int npol,
  int ndat
)
{
  float2* temp = in;
  for (int ichan=0; ichan<nchan; ichan++) {
    for (int ipol=0; ipol<npol; ipol++) {
      for (int idat=0; idat<ndat; idat++) {
        printf("(%f, %f)", (*temp).x, (*temp).y);
        temp++;
      }
      printf("\n");
    }
  }
}

/**
 * Asssumes input data is single channel
 * @method k_overlap_save
 * @param  t_in             [description]
 * @param  t_out            [description]
 * @param  discard_pos      [description]
 * @param  discard_neg      [description]
 * @param  ipart_begin_in   [description]
 * @param  ipart_begin_out  [description]
 * @param  npart            [description]
 * @param  npol             [description]
 * @param  output_nchan     [description]
 * @param  samples_per_part [description]
 * @param  in_ndat          [description]
 * @param  out_ndat         [description]
 */
__global__ void k_overlap_save_one_to_many (
  float2* t_in,
  float2* t_out,
  int discard_pos,
  int discard_neg,
  int ipart_begin_in,
  int ipart_begin_out,
  int npart,
  int npol,
  int nchan,
  int samples_per_part,
  int in_ndat,
  int out_ndat
)
{
  int total_size_nchan = gridDim.x; // for nchan
  int total_size_ndat = blockDim.x; // for ndat
  int total_size_z = blockDim.z * gridDim.z; // for npart and npol
  int npol_incr = total_size_z <= npol ? 1: npol;
  int npart_incr = total_size_z/npol == 0 ? 1: total_size_z/npol;

  int idz = blockIdx.z*blockDim.z + threadIdx.z;

  int step = samples_per_part - (discard_pos + discard_neg);
  // make sure we're not trying to access data that are out of bounds
  if (threadIdx.x > step || blockIdx.x > nchan || idz > npol*npart) {
    return;
  }

  int in_offset;
  int out_offset;

  for (int ochan=blockIdx.x; ochan<nchan; ochan+=total_size_nchan) {
    for (int ipol=idz%npol; ipol<npol; ipol+=npol_incr) {
      for (int ipart=idz/npol; ipart<npart; ipart+=npart_incr) {
        in_offset = ipol*in_ndat + ochan*samples_per_part + (ipart + ipart_begin_in)*samples_per_part;
        out_offset = ochan*npol*out_ndat + ipol*out_ndat + (ipart + ipart_begin_out)*step;
        for (int idat=threadIdx.x; idat<step; idat += total_size_ndat) {
          t_out[out_offset + idat] = t_in[in_offset + idat + discard_pos];
        }
      }
    }
  }
}


__global__ void k_overlap_save_one_to_many_single_part (
  float2* t_in,
  float2* t_out,
  int discard_pos,
  int discard_neg,
  int ipart,
  int npol,
  int nchan,
  int samples_per_part,
  int in_ndat,
  int out_ndat
)
{
  int total_size_x = blockDim.x * gridDim.x; // for ndat
  int total_size_y = blockDim.y * gridDim.y; // for nchan
  int total_size_z = blockDim.z * gridDim.z; // for npol

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  int idz = blockIdx.z*blockDim.z + threadIdx.z;

  int step = samples_per_part - (discard_pos + discard_neg);
  // make sure we're not trying to access data that are out of bounds
  if (idx > step || idy > nchan || idz > npol) {
    return;
  }

  int in_offset;
  int out_offset;

  for (int ochan=idy; ochan<nchan; ochan+=total_size_y) {
    for (int ipol=idz; ipol<npol; ipol+=total_size_z) {
      in_offset = ipol*in_ndat + ochan*samples_per_part;
      out_offset = ochan*npol*out_ndat + ipol*out_ndat + ipart*step;
      for (int idat=idx; idat<step; idat += total_size_x) {
        t_out[out_offset + idat] = t_in[in_offset + idat + discard_pos];
      }
    }
  }
}


__global__ void k_overlap_save (
  float2* t_in,
  float2* t_out,
  int discard_pos,
  int discard_neg,
  int ipart_begin_in,
  int ipart_begin_out,
  int npart,
  int npol,
  int nchan,
  int samples_per_part,
  int in_ndat,
  int out_ndat
)
{
  int total_size_nchan = gridDim.x; // for input nchan
  int total_size_ndat = blockDim.x; // for ndat
  int total_size_z = blockDim.z * gridDim.z; // for npart and npol
  int npol_incr = total_size_z <= npol ? 1: npol;
  int npart_incr = total_size_z/npol == 0 ? 1: total_size_z/npol;

  int idz = blockIdx.z*blockDim.z + threadIdx.z;

  int step = samples_per_part - (discard_pos + discard_neg);
  // make sure we're not trying to access data that are out of bounds
  if (threadIdx.x > step || blockIdx.x > nchan || idz > npol*npart) {
    return;
  }

  int in_offset;
  int out_offset;

  for (int ichan=blockIdx.x; ichan<nchan; ichan+=total_size_nchan) {
    for (int ipol=idz%npol; ipol<npol; ipol+=npol_incr) {
      for (int ipart=idz/npol; ipart<npart; ipart+=npart_incr) {
        in_offset = ichan*npol*in_ndat + ipol*in_ndat + (ipart + ipart_begin_in)*samples_per_part;
        out_offset = ichan*npol*out_ndat + ipol*out_ndat + (ipart + ipart_begin_out)*step;
        for (int idat=threadIdx.x; idat<step; idat += total_size_ndat) {
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
  int ipart_begin_in,
  int ipart_begin_out,
  int npart,
  int npol,
  int nchan,
  int samples_per_part,
  int in_ndat,
  int out_ndat
)
{
  // int total_size_ndat = blockDim.x;
  // int total_size_nchan = gridDim.x;
  int total_size_x = blockDim.x * gridDim.x; // for ndat
  int total_size_y = blockDim.y * gridDim.y; // for nchan
  int total_size_z = blockDim.z * gridDim.z; // for npart and npol
  int npol_incr = total_size_z <= npol ? 1: npol;
  int npart_incr = total_size_z/npol == 0 ? 1: total_size_z/npol;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int idy = blockIdx.y*blockDim.y + threadIdx.y;
  int idz = blockIdx.z*blockDim.z + threadIdx.z;

  int step = samples_per_part - (discard_pos + discard_neg);
  // make sure we're not trying to access data that are out of bounds
  if (idx > samples_per_part || idy > nchan || idz > npol*npart) {
    return;
  }

  int in_offset;
  int out_offset;

  for (int ichan=idy; ichan<nchan; ichan += total_size_y) {
    for (int ipol=idz%npol; ipol<npol; ipol+=npol_incr) {
      for (int ipart=idz/npol; ipart<npart; ipart+=npart_incr) {
        in_offset = ichan*npol*in_ndat + ipol*in_ndat + (ipart + ipart_begin_in)*step;
        out_offset = ichan*npol*out_ndat + ipol*out_ndat + (ipart + ipart_begin_out)*samples_per_part;
        for (int idat=idx; idat<samples_per_part; idat += total_size_x) {
          if (resp == NULL) {
            t_out[out_offset + idat] = t_in[in_offset + idat];
          } else {
            // t_out[out_offset + idat] = cuCmulf(resp[idat], t_in[in_offset + idat]);
            t_out[out_offset + idat].x = resp[idat].x * t_in[in_offset + idat].x;
            t_out[out_offset + idat].y = resp[idat].y * t_in[in_offset + idat].y;
          }
        }
      }
    }
  }
}


__global__ void k_overlap_discard_single_part (
  const  float2* t_in,
  const  float2* resp,
  float2* t_out,
  int discard_pos,
  int discard_neg,
  int ipart,
  int npol,
  int nchan,
  int samples_per_part,
  int in_ndat,
  int out_ndat
)
{
  const int total_size_ndat = blockDim.x;
  const int total_size_nchan = gridDim.x;
  const int total_size_npol = gridDim.y;

  const int dat_idx = threadIdx.x;
  const int chan_idx = blockIdx.x;
  const int pol_idx = blockIdx.y;

  const int step = samples_per_part - (discard_pos + discard_neg);

  // make sure we're not trying to access data that are out of bounds
  if (dat_idx > samples_per_part || chan_idx > nchan || pol_idx > npol) {
    return;
  }

  int in_offset;
  int out_offset;

  for (int ichan=chan_idx; ichan<nchan; ichan += total_size_nchan) {
    for (int ipol=pol_idx; ipol<npol; ipol+=total_size_npol) {
      in_offset = ichan*npol*in_ndat + ipol*in_ndat + ipart*step;
      out_offset = ichan*npol*out_ndat + ipol*out_ndat;
      for (int idat=dat_idx; idat<samples_per_part; idat += total_size_ndat) {
        if (resp == NULL) {
          t_out[out_offset + idat] = t_in[in_offset + idat];
        } else {
          // t_out[out_offset + idat] = cuCmulf(resp[idat], t_in[in_offset + idat]);
          t_out[out_offset + idat].x = resp[idat].x * t_in[in_offset + idat].x;
          t_out[out_offset + idat].y = resp[idat].y * t_in[in_offset + idat].y;
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
  int ipart_begin_in,
  int ipart_begin_out,
  int npart,
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

  // diagnostic
  // int in_total_size = in_nchan * npol * in_ndat;
  // int out_total_size = npol * out_ndat;
  // int resp_total_size = out_samples_per_part;

  for (int ichan=idy; ichan < in_nchan; ichan += total_size_y) {
    for (int ipol=idz%npol; ipol < npol; ipol += npol_incr) {
      for (int ipart=idz/npol; ipart < npart; ipart += npart_incr) {
        in_offset = ichan*npol*in_ndat + ipol*in_ndat + (ipart + ipart_begin_in)*in_samples_per_part;
        out_offset = ipol*out_ndat + (ipart + ipart_begin_out)*out_samples_per_part;

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

          // diagnostic
          // if (out_idx_bot > resp_total_size || out_idx_top > resp_total_size)
          // {
          //   printf("k_response_stitch: overreaching response\n");
          // }
          //
          // if (out_offset + out_idx_bot > out_total_size || out_offset + out_idx_top > out_total_size)
          // {
          //   printf("k_response_stitch: overreaching output\n");
          // }
          //
          // if (in_offset + in_idx_bot > in_total_size || in_offset + in_idx_top > in_total_size)
          // {
          //   printf("k_response_stitch: overreaching input\n");
          // }

          if (response != NULL) {
            f_out[out_offset + out_idx_bot] = cuCmulf(response[out_idx_bot], f_in[in_offset + in_idx_bot]);
            f_out[out_offset + out_idx_top] = cuCmulf(response[out_idx_top], f_in[in_offset + in_idx_top]);
          } else {
            f_out[out_offset + out_idx_bot] = f_in[in_offset + in_idx_bot];
            f_out[out_offset + out_idx_top] = f_in[in_offset + in_idx_top];
          }

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
__global__ void k_response_stitch_single_part (
  const float2* f_in,
  const float2* response,
  float2* f_out,
  int os_discard,
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
  const int total_size_ndat = blockDim.x;
  const int total_size_nchan = gridDim.x;
  const int total_size_npol = gridDim.y;

  const int dat_idx = threadIdx.x;
  const int chan_idx = blockIdx.x;
  const int pol_idx = blockIdx.y;

  const int in_ndat_keep = in_samples_per_part - 2*os_discard;
  const int in_ndat_keep_2 = in_ndat_keep / 2;

  // don't overstep the data
  if (dat_idx > in_ndat_keep_2 || chan_idx > in_nchan || pol_idx > npol) {
    return;
  }

  int in_offset;
  int out_offset;

  int in_idx_bot;
  int in_idx_top;

  int out_idx_bot;
  int out_idx_top;

  // diagnostic
  // int in_total_size = in_nchan * npol * in_ndat;
  // int out_total_size = npol * out_ndat;
  // int resp_total_size = out_samples_per_part;

  for (int ichan=chan_idx; ichan<in_nchan; ichan += total_size_nchan) {
    for (int ipol=pol_idx; ipol<npol; ipol += total_size_npol) {
      in_offset = ichan*npol*in_ndat + ipol*in_ndat;
      out_offset = ipol*out_ndat;

      // in_offset = ipart*npol*in_ndat*in_nchan + ipol*in_ndat*in_nchan + ichan*in_ndat;
      // out_offset = ipart*npol*out_ndat + ipol*out_ndat;
      // std::cerr << "in_offset=" << in_offset << ", out_offset=" << out_offset << std::endl;

      for (int idat=dat_idx; idat<in_ndat_keep_2; idat += total_size_ndat) {
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

        if (response != NULL) {
          f_out[out_offset + out_idx_bot] = cuCmulf(response[out_idx_bot], f_in[in_offset + in_idx_bot]);
          f_out[out_offset + out_idx_top] = cuCmulf(response[out_idx_top], f_in[in_offset + in_idx_top]);
        } else {
          f_out[out_offset + out_idx_bot] = f_in[in_offset + in_idx_bot];
          f_out[out_offset + out_idx_top] = f_in[in_offset + in_idx_top];
        }

        if (! pfb_all_chan && pfb_dc_chan && ichan == 0) {
          f_out[out_offset + out_idx_bot].x = 0.0;
          f_out[out_offset + out_idx_bot].y = 0.0;
        }
      }
    }
  }
}


// __global__ void k_response_stitch_single_part_op (
//   const float2* f_in,
//   const float2* response,
//   float2* f_out,
//   int os_discard,
//   // int npol,
//   // int in_nchan,
//   int in_ndat,
//   int out_ndat,
//   int in_samples_per_part,
//   int out_samples_per_part,
//   bool pfb_dc_chan,
//   bool pfb_all_chan
// )
// {
//   const int total_size_x = blockDim.x * gridDim.x; // for idat
//   // const int in_nchan = blockDim.y; // for ichan
//   const int npol = blockDim.z; // for ipol
//
//   const int idx = blockIdx.x*blockDim.x + threadIdx.x;
//   // int idy = blockIdx.y*blockDim.y + threadIdx.y;
//   // int idz = blockIdx.z*blockDim.z + threadIdx.z;
//
//   // int idy = blockIdx.y;
//   // int idz = blockIdx.z;
//
//
//   const int in_ndat_keep = in_samples_per_part - 2*os_discard;
//   const int in_ndat_keep_2 = in_ndat_keep / 2;
//
//   // don't overstep the data
//   if (idx > in_ndat_keep_2) { // || blockIdx.y > blockDim.y || blockIdx.z > npol) {
//     return;
//   }
//
//   int in_idx_bot;
//   int in_idx_top;
//
//   int out_idx_bot;
//   int out_idx_top;
//
//   // diagnostic
//   // int in_total_size = in_nchan * npol * in_ndat;
//   // int out_total_size = npol * out_ndat;
//   // int resp_total_size = out_samples_per_part;
//
//   // for (int ichan=idy; ichan < in_nchan; ichan += total_size_y) {
//   //   for (int ipol=idz; ipol < npol; ipol += total_size_z) {
//   int in_offset = blockIdx.y*npol*in_ndat + blockIdx.z*in_ndat;
//   int out_offset = blockIdx.z*out_ndat;
//
//   // in_offset = ipart*npol*in_ndat*in_nchan + blockIdx.z*in_ndat*in_nchan + blockIdx.y*in_ndat;
//   // out_offset = ipart*npol*out_ndat + blockIdx.z*out_ndat;
//   // std::cerr << "in_offset=" << in_offset << ", out_offset=" << out_offset << std::endl;
//
//   for (int idat=idx; idat<in_ndat_keep_2; idat += total_size_x) {
//     in_idx_top = idat;
//     in_idx_bot = in_idx_top + (in_samples_per_part - in_ndat_keep_2);
//
//     out_idx_bot = idat + in_ndat_keep*blockIdx.y;
//     out_idx_top = out_idx_bot + in_ndat_keep_2;
//
//     if (pfb_dc_chan) {
//       if (blockIdx.y == 0) {
//         out_idx_top = idat;
//         out_idx_bot = idat + (out_samples_per_part - in_ndat_keep_2);
//       } else {
//         out_idx_bot = idat + in_ndat_keep*(blockIdx.y-1) + in_ndat_keep_2;
//         out_idx_top = out_idx_bot + in_ndat_keep_2;
//       }
//     }
//
//     if (response != NULL) {
//       f_out[out_offset + out_idx_bot] = cuCmulf(response[out_idx_bot], f_in[in_offset + in_idx_bot]);
//       f_out[out_offset + out_idx_top] = cuCmulf(response[out_idx_top], f_in[in_offset + in_idx_top]);
//     } else {
//       f_out[out_offset + out_idx_bot] = f_in[in_offset + in_idx_bot];
//       f_out[out_offset + out_idx_top] = f_in[in_offset + in_idx_top];
//     }
//
//     if (! pfb_all_chan && pfb_dc_chan && blockIdx.y == 0) {
//       f_out[out_offset + out_idx_bot].x = 0.0;
//       f_out[out_offset + out_idx_bot].y = 0.0;
//     }
//   }
// }



CUDA::InverseFilterbankEngineCUDA::InverseFilterbankEngineCUDA (cudaStream_t _stream)
{
  stream = _stream;

  input_fft_length = 0;
  forward_fft_plan_setup = false;
  backward_fft_plan_setup = false;
  response = NULL;
  fft_window = NULL;

  d_scratch = NULL;
  d_input_overlap_discard = NULL;
  d_stitching = NULL;

  d_response = NULL;
  d_fft_window = NULL;

  pfb_dc_chan = 0;
  pfb_all_chan = 0;
  verbose = dsp::Observation::verbose;
  report = false;

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
  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::~InverseFilterbankEngineCUDA" << std::endl;
  }
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

  // float2* cuda_scratches[] = {d_scratch, d_response, d_fft_window};
  // int n_scratches = sizeof(cuda_scratches) / sizeof(cuda_scratches[0]);
  //
  // for (int i=0; i<n_scratches; i++)
  // {
  //   if (cuda_scratches[i] != NULL) {
  //     cudaError_t error = cudaFree (cuda_scratches[i]);
  //     if (error != cudaSuccess) {
  //       if (verbose) {
  //         std::cerr << "CUDA::InverseFilterbankEngineCUDA::~InverseFilterbankEngineCUDA: failed to deallocate memory: " << cudaGetErrorString (error) << std::endl;
  //       }
  //       throw Error (FailedCall, "CUDA::InverseFilterbankEngineCUDA::~InverseFilterbankEngineCUDA",
  //                    "cudaFree(%xu): %s", &cuda_scratches[i],
  //                    cudaGetErrorString (error));
  //     }
  //   }
  // }
}

std::vector<cufftResult> CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan (
  unsigned _input_fft_length,
  unsigned _howmany,
  cufftType _type_forward
)
{
  if (verbose)
  {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan("
      << _input_fft_length << ", " << _howmany << ", " << _type_forward << ")" << std::endl;
  }
  // setup forward batched plan
  int rank = 1; // 1D transform
  int n[] = {(int) _input_fft_length}; /* 1d transforms of length _input_fft_length */
  int howmany = _howmany;
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

  if (result != CUFFT_SUCCESS) {
     if (verbose) {
       std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan: couldn't set up forward FFT plan: " << result << std::endl;
     }

    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan",
                      "cufftPlanMany(forward)");
  }
  result = cufftSetStream (forward, stream);
  results.push_back(result);

  if (result != CUFFT_SUCCESS) {
     if (verbose) {
       std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan: couldn't set stream: " << result << std::endl;
     }
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_forward_fft_plan",
          "cufftSetStream(forward)");
  }
  forward_fft_plan_setup = true;
  return results;
}

std::vector<cufftResult> CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan (
  unsigned _output_fft_length,
  unsigned _howmany
)
{
  if (verbose)
  {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan("
      << _output_fft_length << ", " << _howmany << ")" << std::endl;
  }
  // setup forward batched plan
  int rank = 1; // 1D transform
  int n[] = {(int) _output_fft_length}; /* 1d transforms of length _output_fft_length */
  int howmany = _howmany;
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

  if (result != CUFFT_SUCCESS) {
     if (verbose) {
       std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan: couldn't set up backward fft plan: " << result << std::endl;
     }
     throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan",
                      "cufftPlanMany(backward)");
  }
  result = cufftSetStream (backward, stream);
  results.push_back(result);
  if (result != CUFFT_SUCCESS) {
     if (verbose) {
       std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan: couldn't set stream: " << result << std::endl;
     }
    throw CUFFTError (result, "CUDA::InverseFilterbankEngineCUDA::setup_backward_fft_plan",
                      "cufftSetStream(backward)");
   }
  backward_fft_plan_setup = true;

  return results;
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
  const Rational& os_factor = filterbank->get_oversampling_factor();
  unsigned _input_fft_length = filterbank->get_input_fft_length();
  unsigned _output_fft_length = filterbank->get_output_fft_length();
  unsigned _input_discard_pos = filterbank->get_input_discard_pos();
  unsigned _input_discard_neg = filterbank->get_input_discard_neg();
  unsigned _output_discard_pos = filterbank->get_output_discard_pos();
  unsigned _output_discard_neg = filterbank->get_output_discard_neg();
  bool _pfb_dc_chan = filterbank->get_pfb_dc_chan();
  bool _pfb_all_chan = filterbank->get_pfb_all_chan();

  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup: "
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
      << _pfb_all_chan << std::endl;
  }
  type_forward = CUFFT_C2C;
  n_per_sample = 1;

  if (input->get_state() == Signal::Nyquist) {
    type_forward = CUFFT_R2C;
    n_per_sample = 1;
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
  if (input_os_discard % 2 != 0)
  {
    throw "CUDA::InverseFilterbankEngineCUDA::setup: input_os_discard must be divisible by two";
  }

  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup:"
      << " input_os_keep=" << input_os_keep
      << " input_os_discard=" << input_os_discard
      << std::endl;
  }

  if (filterbank->has_response())
  {
    response = filterbank->get_response();
  }

  if (filterbank->has_apodization())
  {
    fft_window = filterbank->get_apodization();
  }

  setup_forward_fft_plan(
    input_fft_length, input_nchan*input_npol, type_forward
  );

  setup_backward_fft_plan(
    output_fft_length, output_nchan*input_npol
  );

  cudaError_t error;
  // need device memory for response
  if (response) {
    if (verbose) {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup: copying response from host to device" << std::endl;
    }
    unsigned response_size = response->get_nchan() * response->get_ndat() * sizeof(cufftComplex);
    const float* response_kernel = response->get_datptr(0, 0);
    cudaMalloc((void**) &d_response, response_size);
    // d_response = (float2*) memory->do_allocate(response_size);
    if (stream) {
      error = cudaMemcpyAsync(
        d_response, response_kernel, response_size, cudaMemcpyHostToDevice, stream);
    } else {
      error = cudaMemcpy(
        d_response, response_kernel, response_size, cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess)
    {
      throw Error (InvalidState, "CUDA::InverseFilterbankEngineCUDA::setup",
       "could not copy dedispersion kernel to device");
    }
  }
  // need device memory for apodization
  if (fft_window) {
    if (verbose) {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup: copying fft window from host to device" << std::endl;
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup: fft_window.get_type() "
        << fft_window->get_type() << std::endl;
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup: fft_window.get_ndim() "
        << fft_window->get_ndim() << std::endl;
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup: fft_window.get_ndat() "
        << fft_window->get_ndat() << std::endl;
    }
    unsigned fft_window_size = fft_window->get_nchan() * fft_window->get_ndat() * sizeof(cufftComplex);
    const float* fft_window_kernel = fft_window->get_datptr(0, 0);
    cudaMalloc((void**) &d_fft_window, fft_window_size);
    // d_fft_window = (float2*) memory->do_allocate(fft_window_size);
    if (stream) {
      error = cudaMemcpyAsync(
        d_fft_window, fft_window_kernel, fft_window_size, cudaMemcpyHostToDevice, stream);
    } else {
      error = cudaMemcpy(
        d_fft_window, fft_window_kernel, fft_window_size, cudaMemcpyHostToDevice);
    }
    if (error != cudaSuccess)
    {
      throw Error (InvalidState, "CUDA::InverseFilterbankEngineCUDA::setup",
       "could not copy FFT window response to device");
    }
  }

  d_input_overlap_discard_samples = input_npol*input_nchan*input_fft_length;
  d_stitching_samples = input_npol*output_nchan*output_fft_length;

  // we multiply by two because the device scratch space point to float2 arrays,
  // and the Scratch object allocates in float.
  total_scratch_needed = 2*(d_input_overlap_discard_samples + d_stitching_samples);

  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::setup:"
      << " d_input_overlap_discard_samples=" << d_input_overlap_discard_samples
      << " d_stitching_samples=" << d_stitching_samples
      << " total_scratch_needed=" << total_scratch_needed
      << std::endl;
  }

}


void CUDA::InverseFilterbankEngineCUDA::set_scratch (float* _scratch)
{
  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::set_scratch(" << _scratch << ")" << std::endl;
  }
  d_scratch = (float2*) _scratch;
  d_input_overlap_discard = d_scratch;
  d_stitching = d_scratch + d_input_overlap_discard_samples;
  // check_error ("CUDA::InverseFilterbankEngineCUDA::set_scratch");
}

void CUDA::InverseFilterbankEngineCUDA::perform (
  const dsp::TimeSeries* in,
  dsp::TimeSeries* out,
  uint64_t npart,
  uint64_t in_step,
  uint64_t out_step
)
{
  perform(in, out, NULL, npart, in_step, out_step);
}

void CUDA::InverseFilterbankEngineCUDA::perform (
  const dsp::TimeSeries* in,
  dsp::TimeSeries* out,
  dsp::TimeSeries* zero_DM_out,
  uint64_t npart,
  uint64_t in_step,
  uint64_t out_step
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
  unsigned input_stride = in->get_stride() / in->get_ndim();
  unsigned output_stride = out->get_stride() / out->get_ndim();

  unsigned _input_stride = in->internal_get_subsize();
  unsigned _output_stride = out->internal_get_subsize();

  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: in shape=("
      << in->get_nchan() << ", " << in->get_npol() << ", " << in->get_ndat() << ")" << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: out shape=("
      << out->get_nchan() << ", " << out->get_npol() << ", " << out->get_ndat() << ")" << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform:"
      << " input_nchan=" << input_nchan
      << " input_npol=" << input_npol
      << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform:"
      << " report=" << report
      << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform:"
      << " stream=" << stream
      << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: input_stride=" << input_stride << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: _input_stride=" << _input_stride << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: output_stride=" << output_stride << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: _output_stride=" << _output_stride << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: in.internal_get_size()=" << in->internal_get_size() << std::endl;
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: out.internal_get_size()=" << out->internal_get_size() << std::endl;
  }


  const float* in_ptr = in->get_datptr(0, 0);
  float* out_ptr = out->get_datptr(0, 0);

  int nthreads_k_overlap_discard = (input_fft_length <= 1024) ? input_fft_length: 1024;

  int nthreads_k_response_stitch = (input_fft_length - input_os_discard) / 2;
  nthreads_k_response_stitch = (nthreads_k_response_stitch <= 1024) ? nthreads_k_response_stitch: 1024;

  int nthreads_k_overlap_save = (output_fft_length <= 1024) ? output_fft_length: 1024;

  dim3 grid (input_nchan, input_npol, 1);
  dim3 threads (nthreads_k_overlap_discard, 1, 1);

  int k_response_stitch_in_samples_per_part = input_fft_length;
  int k_response_stitch_out_samples_per_part = output_nchan * output_fft_length;

  int k_response_stitch_in_ndat = input_fft_length;
  int k_response_stitch_out_ndat = output_nchan * output_fft_length;

  // if (verbose) {
  //   std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: grid=("
  //     << grid.x << ", " << grid.y << ", " << grid.z << ")" << std::endl;
  //   std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: threads=("
  //     << threads.x << ", " << threads.y << ", " << threads.z << ")" << std::endl;
  // }
  for (unsigned ipart=0; ipart<npart; ipart++)
  {
    if (verbose)
    {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: part " << ipart << "/" << npart << std::endl;
    }
    if (type_forward == CUFFT_R2C)
    {
      if (verbose)
      {
        std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: R2C applying overlap discard kernel" << std::endl;
      }
      throw "Currently incompatible with R2C";
    }
    else
    {
      if (verbose) {
        std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: applying overlap discard kernel" << std::endl;
      }

      grid.x = input_nchan;
      grid.y = input_npol;
      grid.z = 1;

      threads.x = nthreads_k_overlap_discard;

      // k_overlap_discard<<<grid, threads, 0, stream>>> (
      //   (cufftComplex*) in_ptr,
      //   d_fft_window,
      //   (cufftComplex*) d_input_overlap_discard,
      //   input_discard_pos,
      //   input_discard_neg,
      //   ipart,
      //   0, 1,
      //   input_npol,
      //   input_nchan,
      //   input_fft_length,
      //   input_stride,
      //   input_fft_length
      // );
      k_overlap_discard_single_part<<<grid, threads, 0, stream>>> (
        (cufftComplex*) in_ptr,
        d_fft_window,
        (cufftComplex*) d_input_overlap_discard,
        input_discard_pos,
        input_discard_neg,
        ipart,
        input_npol,
        input_nchan,
        input_fft_length,
        input_stride,
        input_fft_length
      );

      if (report) {
        check_error("CUDA::InverseFilterbankEngineCUDA::perform: k_overlap_discard");
        reporter.emit("fft_window",
          (float*) d_input_overlap_discard,
          input_nchan, input_npol, input_fft_length, 2
        );
      }
      if (verbose) {
        std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: applying forward FFT" << std::endl;
      }
       cufftExecC2C(
         forward,
          (cufftComplex*) d_input_overlap_discard,
         (cufftComplex*) d_input_overlap_discard,
         CUFFT_FORWARD
       );

       if (report) {
         check_error("CUDA::InverseFilterbankEngineCUDA::perform: fft");
         reporter.emit("fft",
           (float*) d_input_overlap_discard,
           input_nchan, input_npol, input_fft_length, 2
         );
       }
    }

    if (verbose)
    {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: applying response stitch kernel" << std::endl;
    }

    grid.x = input_nchan;
    grid.y = input_npol;
    grid.z = 1;

    threads.x = nthreads_k_response_stitch;

    // k_response_stitch<<<grid, threads, 0, stream>>>(
    //   (float2*) d_input_overlap_discard,
    //   d_response,
    //   (float2*) d_stitching,
    //   input_os_discard/2,
    //   0, 0, 1,
    //   input_npol, input_nchan,
    //   k_response_stitch_in_ndat,
    //   k_response_stitch_out_ndat,
    //   k_response_stitch_in_samples_per_part,
    //   k_response_stitch_out_samples_per_part,
    //   pfb_dc_chan, pfb_all_chan
    // );
    k_response_stitch_single_part<<<grid, threads, 0, stream>>>(
      (float2*) d_input_overlap_discard,
      d_response,
      (float2*) d_stitching,
      input_os_discard/2,
      input_npol, input_nchan,
      k_response_stitch_in_ndat,
      k_response_stitch_out_ndat,
      k_response_stitch_in_samples_per_part,
      k_response_stitch_out_samples_per_part,
      pfb_dc_chan, pfb_all_chan
    );

    if (report) {
      check_error("CUDA::InverseFilterbankEngineCUDA::perform: k_response_stitch");
      reporter.emit("response_stitch", (float*) d_stitching,
        1, input_npol, output_fft_length*output_nchan, 2);
    }

    if (verbose)
    {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: applying inverse FFT" << std::endl;
    }
    cufftExecC2C (
      backward,
      (cufftComplex*) d_stitching,
      (cufftComplex*) d_stitching,
      CUFFT_INVERSE
    );

    if (report) {
      check_error("CUDA::InverseFilterbankEngineCUDA::perform: ifft");
      reporter.emit("ifft", (float*) d_stitching,
        1, input_npol, output_fft_length*output_nchan, 2);
    }

    if (verbose)
    {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: applying overlap save kernel" << std::endl;
    }

    grid.x = (output_fft_length <= 1024) ? 1: output_fft_length/1024;
    grid.y = output_nchan;
    grid.z = input_npol;

    threads.x = nthreads_k_overlap_save;

    // k_overlap_save_one_to_many<<<grid, threads, 0, stream>>>(
    // // k_overlap_save_one_to_many<<<1, 1, 0, stream>>>(
    //   (float2*) d_stitching,
    //   (float2*) out_ptr,
    //   output_discard_pos,
    //   output_discard_neg,
    //   0, ipart, 1,
    //   input_npol,
    //   output_nchan,
    //   output_fft_length,
    //   output_fft_length*output_nchan,
    //   output_stride
    // );


    k_overlap_save_one_to_many_single_part<<<grid, threads, 0, stream>>>(
      (float2*) d_stitching,
      (float2*) out_ptr,
      output_discard_pos,
      output_discard_neg,
      ipart,
      input_npol,
      output_nchan,
      output_fft_length,
      output_fft_length*output_nchan,
      output_stride
    );
    if (report) {
      check_error("CUDA::InverseFilterbankEngineCUDA::perform: k_overlap_save_one_to_many");
    }
  }
  if (dsp::Operation::record_time || dsp::Operation::verbose || record_time ) {
    if (verbose) {
      std::cerr << "CUDA::InverseFilterbankEngineCUDA::perform: recording timing" << std::endl;
    }
    if (stream) {
      check_error_stream("CUDA::InverseFilterbankEngineCUDA::perform", stream);
    } else {
      check_error("CUDA::InverseFilterbankEngineCUDA::perform");
    }
  }
}

void CUDA::InverseFilterbankEngineCUDA::finish ()
{
  if (verbose) {
    std::cerr << "CUDA::InverseFilterbankEngineCUDA::finish" << std::endl;
  }
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
  int nthreads = in_ndat <= 1024 ? in_ndat: 1024;

  dim3 grid (1, nchan, npart*npol);
  dim3 threads (nthreads, 1, 1);

  k_response_stitch<<<grid, threads>>>(
    in_device, resp_device, out_device, os_discard, 0, 0, npart,
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
    in_device, apod_device, out_device, discard, discard, 0, 0, npart, npol, nchan, ndat, in_ndat, out_ndat);
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
    in_device, NULL, out_device, discard, discard, 0, 0, npart, npol, nchan, ndat, in_ndat, out_ndat);

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

  dim3 grid (nchan, 1, npol*npart);
  dim3 threads (1024, 1, 1);

  // std::cerr << grid.x << " " << grid.y << " " << grid.z << std::endl;
  // std::cerr << threads.x << " " << threads.y << " " << threads.z << std::endl;

  k_overlap_save<<<grid, threads>>>(
    in_device, out_device, discard, discard,
    0, 0, npart, npol, nchan, ndat, in_ndat, out_ndat);

  check_error( "CUDA::InverseFilterbankEngineCUDA::apply_k_overlap_save" );

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
