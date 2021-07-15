//-*-C++-*-

/***************************************************************************
 *
 *   Copyright (C) 2018 by Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FITSDigitizerCUDA.h"
#include "dsp/TimeSeries.h"

#include "Error.h"
#include "debug.h"

#include <cuda_runtime.h>
#include <unistd.h>

#define FULLMASK 0xFFFFFFFF

using namespace std;

void check_error_stream (const char*, cudaStream_t);

CUDA::FITSDigitizerEngine::FITSDigitizerEngine (cudaStream_t _stream)
{
  stream = _stream;

  freq_totalsq = NULL;
  freq_total = NULL;
  freq_total_size = 0;

  d_scale = h_scale = NULL;
  d_offset = h_offset = NULL;
  scale_offset_size = 0;

  mapping = NULL;
  mapping_size = 0;

  scratch = NULL;
}

CUDA::FITSDigitizerEngine::~FITSDigitizerEngine ()
{
  if (freq_totalsq)
    cudaFree (freq_totalsq);
  freq_totalsq = NULL;

  if (freq_total)
    cudaFree (freq_total);
  freq_total = NULL;

  if (d_offset)
    cudaFree (d_offset);
  d_offset = NULL;

  if (h_offset)
    cudaFreeHost (h_offset);
  h_offset = NULL;

  if (d_scale)
    cudaFree (d_scale);
  d_scale = NULL;

  if (h_scale)
    cudaFreeHost (h_scale);
  h_scale = NULL;
}

void CUDA::FITSDigitizerEngine::set_scratch (dsp::Scratch * _scratch)
{
  scratch = _scratch;
}

void CUDA::FITSDigitizerEngine::set_rescale_nblock (const dsp::TimeSeries * input, unsigned nblock)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::FITSDigitizerEngine::set_rescale_nblock nblock=" << nblock << endl;
  rescale_nblock = nblock;
  
  unsigned nchan = input->get_nchan();
  unsigned npol = input->get_npol();

  if (dsp::Operation::verbose)
    cerr << "CUDA::FITSDigitizerEngine::set_rescale_nblock nchan=" << nchan
          << " npol=" << npol << endl;

  size_t req_size = npol * nchan * rescale_nblock * sizeof(float);
  if (req_size > freq_total_size)
  {
    if (freq_totalsq)
      cudaFree(freq_totalsq);
    if (freq_total)
      cudaFree(freq_total);

    freq_total_size = req_size;
    cudaMalloc (&freq_totalsq, req_size);
    cudaMalloc (&freq_total, req_size);
  }

  req_size = npol * nchan * sizeof(float);
  if (req_size > scale_offset_size)
  {
    if (d_scale)
      cudaFree(d_scale);
    if (h_scale)
      cudaFreeHost(h_scale);

    if (d_offset)
      cudaFree(d_offset);
    if (h_offset)
      cudaFreeHost(h_offset);

    scale_offset_size = req_size;
    cudaMalloc (&d_scale, scale_offset_size);
    cudaMallocHost (&h_scale, scale_offset_size);
    cudaMalloc (&d_offset, scale_offset_size);
    cudaMallocHost (&h_offset, scale_offset_size);
  }
}

void CUDA::FITSDigitizerEngine::set_mapping (const dsp::TimeSeries * input, 
                                             dsp::ChannelSort& channel)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::FITSDigitizerEngine::set_mapping" << endl;
  unsigned nchan = input->get_nchan();
  size_t req_size = nchan * sizeof(unsigned);
  if (req_size > mapping_size)
  {
    if (mapping)
      cudaFree(mapping);
    mapping_size = req_size;
    cudaMalloc (&mapping, mapping_size);
  }

  // build the mapping on the host buffer (output to input channels)
  unsigned * mapping_host = NULL;
  cudaMallocHost ((void **) &mapping_host, req_size);
  for (unsigned ichan=0; ichan<nchan; ichan++)
  {
    mapping_host[ichan] = channel(ichan);
  }

  // copy the mapping to device buffer
  cudaMemcpyAsync(mapping, mapping_host, mapping_size, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
  cudaFreeHost(mapping_host);
}

// compute a sum of a float across a warp
__inline__ __device__ float fde_warp_reduce_sum(float val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
  {
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__>= 9)
    val += __shfl_down_sync(FULLMASK, val, offset);
#else
    val += __shfl_down(val, offset);
#endif
#endif
  }
  return val;
}

// compute a sum of a float across a block
__inline__ __device__ float fde_block_reduce_sum (float val)
{
  // shared mem for 32 partial sums
  __shared__ float shared[32];

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  // each warp performs partial reduction
  val = fde_warp_reduce_sum(val);

  // write reduced value to shared memory
  if (lane==0)
    shared[wid] = val;

  // wait for all partial reductions
  __syncthreads();

  // read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  // final reduce within first warp
  if (wid==0) 
    val = fde_warp_reduce_sum(val);

  return val;
}


// compute sum and squared sum
// each warp processes 1 channel/pol
__global__ void fde_tsum_tfp (const float * in, float *ft, float * fts, 
                              float  * offset, float * scale,
                              uint64_t ndat, unsigned nchanpol, 
                              const float recip)
{
  const unsigned warp_idx = threadIdx.x % warpSize;
  const unsigned warp_num = threadIdx.x / warpSize;

  // each warp processes 1 channel, each block of 1024 processes 32 channels
  unsigned ichanpol = (blockIdx.x * warpSize) + warp_num;

  // the sample offset for this thread
  uint64_t idx = (warp_idx * nchanpol) + ichanpol;

  float ft_thread = 0;
  float fts_thread = 0;

  if (ichanpol < nchanpol)
  {
    const uint64_t warp_stride = nchanpol * warpSize;

    // process all of the samples for this chan/pol
    for (uint64_t idat=warp_idx; idat<ndat; idat+=warpSize)
    {
      const float in_val = in[idx];
      ft_thread  += in_val;
      fts_thread += (in_val * in_val);
      idx += warp_stride;
    }

    // now reduce across the warp
    ft_thread = fde_warp_reduce_sum (ft_thread);
    __syncthreads();
    fts_thread = fde_warp_reduce_sum (fts_thread);

    // store ft, fts, offset and scale in FP order
    if (warp_idx == 0)
    {
      ft[ichanpol] = ft_thread;
      fts[ichanpol] = fts_thread;

      const float offset_chanpol = ft_thread * recip;
      offset[ichanpol] = offset_chanpol;
      scale[ichanpol] = sqrt (fts_thread*recip - offset_chanpol*offset_chanpol);
    }
  }
}


// each block processes 1 channel/pol
__global__ void fde_tsum_fpt (const float * in, uint64_t chanpol_stride,
                              float *ft, float * fts, 
                              float * offset, float * scale,
                              uint64_t ndat, float recip)
{
  const unsigned ichanpol = blockIdx.x;
  const uint64_t chanpol_offset = ichanpol * chanpol_stride;

  float ft_thread = 0;
  float fts_thread = 0;

  const float * in_ptr = in + chanpol_offset;

  for (uint64_t idat=threadIdx.x; idat<ndat; idat+=blockDim.x)
  {
    const float in_val = in_ptr[idat];
    ft_thread  += in_val;
    fts_thread += (in_val * in_val);
  }

  // sum across block
  ft_thread  = fde_block_reduce_sum(ft_thread);

  // force a sync here so since shared memory is shared in the 2 reductions
  __syncthreads();

  // sum across block
  fts_thread = fde_block_reduce_sum(fts_thread);

  if (threadIdx.x == 0)
  {
    // FP ordered
    ft[ichanpol] = ft_thread;
    fts[ichanpol] = fts_thread;

    const float offset_chanpol = ft_thread * recip;

    offset[ichanpol] = offset_chanpol;
    scale[ichanpol] = sqrt (fts_thread*recip - offset_chanpol*offset_chanpol);
  }
}


//! 
void CUDA::FITSDigitizerEngine::measure_scale (const dsp::TimeSeries *in, unsigned rescale_nsamp)
{
  unsigned ndim = in->get_ndim();
  unsigned nchan = in->get_nchan();
  unsigned npol = in->get_npol();

  unsigned nchanpol = nchan * npol;
  if (dsp::Operation::verbose)
    cerr << "CUDA::FITSDigitizerEngine::measure_scale ndim=" << ndim 
         << " nchan=" << nchan << " npol=" << npol << " ndat=" << rescale_nsamp
         << endl;

  if (ndim != 1)
    throw Error (InvalidState, "CUDA::FITSDigitizerEngine::measure_scale",
                "detected data expected ndim=1");

  if (rescale_nblock != 1)
    throw Error (InvalidState, "CUDA::FITSDigitizerEngine::measure_scale",
                "rescale_nblock must be == 1, for now");

  // zero the storage arrays
  cudaMemsetAsync(freq_total, 0, freq_total_size, stream);
  cudaMemsetAsync(freq_totalsq, 0, freq_total_size, stream);
  if (dsp::Operation::record_time || dsp::Operation::verbose)
    check_error_stream( "CUDA::FITSDigitizerEngine::measure_scale cudaMemsetAsync", stream );

  const float recip = 1.0 / float(rescale_nsamp);
  unsigned nthreads = 1024;
  unsigned warpSize = 32;

  // for each chanpol measure sum(samples) and sum(samples^2)
  switch (in->get_order())
  {
    case dsp::TimeSeries::OrderTFP:
    {
      const float * in_ptr = in->get_dattfp();
      unsigned nblocks = nchanpol / warpSize;
      if (nchanpol % warpSize != 0)
      {
        cerr << "Warning nchanpol " << nchanpol << " not a multiple of 32" << endl;
        nblocks++;
      }

#ifdef _DEBUG
      cerr << "CUDA::FITSDigitizerEngine::measure_scale fde_tsum_ftp nblocks="
           << nblocks << " nthreads=" << nthreads << " ndat=" << rescale_nsamp 
           << " recip=" << recip << endl;
#endif
      fde_tsum_tfp<<<nblocks, nthreads, 0, stream>>>(in_ptr, 
                                                     freq_total, freq_totalsq, 
                                                     d_offset, d_scale,
                                                     rescale_nsamp, nchanpol, recip);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream( "CUDA::FITSDigitizerEngine::measure_scale fde_tsum_tfp", stream );
      break; 
    }

    case dsp::TimeSeries::OrderFPT:
    {
      unsigned nblocks = nchanpol;
      const float * first_chanpol = in->get_datptr(0,0);
      uint64_t chanpol_stride = 0;
      if (npol == 1 && nchan > 1)
      {
        const float * next_chanpol = in->get_datptr (1, 0);
        chanpol_stride = next_chanpol - first_chanpol;
      }
      else if (npol > 1)
      {
        const float * next_chanpol = in->get_datptr (0, 1);
        chanpol_stride = next_chanpol - first_chanpol;
      }
#ifdef _DEBUG
      cerr << "CUDA::FITSDigitizerEngine::measure_scale nblocks=" << nblocks << " nthreads=" << nthreads 
           << " chanpol_stride=" << chanpol_stride << " ndat=" << rescale_nsamp << " recip=" << recip << endl;
#endif
      fde_tsum_fpt<<<nblocks, nthreads, 0, stream>>>(first_chanpol, chanpol_stride,
                                                     freq_total, freq_totalsq, 
                                                     d_offset, d_scale, 
                                                     rescale_nsamp, recip);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream( "CUDA::FITSDigitizerEngine::measure_scale fde_tsum_fpt", stream );
      break;
    }
  
    default:
      throw Error (InvalidState, "CUDA::FITSDigitizerEngine::measure_scale",
        "Requires data in TFP or FPT order");
  }
}

// rescale, and reorder the input data from TFP to TPF
__global__ void fde_reorder_tfp (const float * in, const unsigned * mapping, 
                      float * offset, float * scale, int * output,
                      float digi_scale, float digi_mean,
                      unsigned nchan, unsigned npol)
{
  // each block will process 1 sample
  const uint64_t dat_offset = blockIdx.x * nchan * npol;

  // process all the channels in this block
  for (unsigned ichan=threadIdx.x; ichan<nchan; ichan+=blockDim.x)
  {
    // input and output offsets
    uint64_t idx = dat_offset + (ichan * npol);
    uint64_t odx = dat_offset + (mapping[ichan] * npol);

    for (unsigned ipol=0; ipol<npol; ipol++)
    {
      const unsigned ichanpol = ichan * npol + ipol;

      // convert to 0 mean and unit variance
      float val = (in[idx] - offset[ichanpol]) / scale[ichanpol];

      // now rescale as an integer,
      output[odx] = int((val * digi_scale) + digi_mean + 0.5);

      idx++;
      odx++;
    }
  }
}

// FPT to TPF order. rescale, and reorder the input data
__global__ void fde_reorder_fpt (const float * in, uint64_t chanpol_stride,
         const unsigned * mapping,
         float * offset, float * scale, int * output,
         float digi_scale, float digi_mean, uint64_t ndat,
         unsigned nchan, unsigned npol)
{
  extern __shared__ int fde_reorder_fpt_shm[];

  const unsigned warp_num = threadIdx.x / warpSize;
  const unsigned warp_idx = threadIdx.x % warpSize;

  const unsigned ipol = blockIdx.y;

  // each block processing 32 time samples for all channels
  const uint64_t idat = (blockIdx.x * 32) + warp_idx;

  unsigned ochan_block = 0;

  // each warp reads 32-consecutive time samples for 1 output channel
  for (unsigned ochan=warp_num; ochan<nchan; ochan+=warpSize)
  {
    if (idat < ndat)
    {
      const unsigned ichanpol = (mapping[ochan] * npol) + ipol; 

      // read the input value, subtrace the offset divide by scale
      const float val = (in[(ichanpol * chanpol_stride) + idat] - offset[ichanpol]) / scale[ichanpol];

      // save to SHM in FT order after converting to digisation scale
      fde_reorder_fpt_shm[(warp_num * 32) + warp_idx] = int((val * digi_scale) + digi_mean + 0.5);
    }

    // now we have 32 time samples for 32 channels in SHM [32*32*4] = 4096 bytes
    __syncthreads();

    // each warp writes out a time sample for 32 channels
    uint64_t odat = (blockIdx.x * 32) + warp_num;
    if (odat < ndat)
    {
      // output is ordered in TPF order [FITS convention]
      //                    odat * odat_stride   +  opol * pol_stride + ochan* ochan_stride
      const uint64_t odx = (odat * nchan * npol) + (ipol * nchan)     + (ochan_block + warp_idx);

      //              ichan  * nsamp + idat
      const unsigned sdx = (warp_idx * 32) + warp_num;

      output[odx] = fde_reorder_fpt_shm[sdx];
    }

    // synchronize to ensure all threads have read their shared memory before iterating the loop again
    __syncthreads();

    ochan_block += warpSize;
  }
}


// quantise a TFP ordered signal to 1b
__global__ void fde_digitize_nbit (int * input, unsigned char * output, uint64_t nval,
                        unsigned nbit, int digi_min, int digi_max)
{
  unsigned samples_per_byte = 8 / nbit;

  uint64_t odx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t idx = odx * samples_per_byte;

  if (odx >= nval)
    return;

  unsigned char out = 0;

  // each thread writes and output char 
  for (unsigned i=0; i<samples_per_byte; i++)
  {
    int result = input[idx];

    // clip the result
    result = max(result, digi_min);
    result = min(result, digi_max);

    // earlier samples in the more significant bits
    unsigned bit_shift = (samples_per_byte - i - 1) * nbit;

    // logical or together, noting that the later samples are 
    // in the more significant bits
    out |= ((unsigned char) result) << bit_shift;
    
    idx++;
  }

  output[odx] = out;
}

__global__ void fde_digitize_8bit (int * in, unsigned char * out,
                        uint64_t nval, int digi_min, int digi_max)
{
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nval)
    return;

  // clip the result
  int result = max(in[idx], digi_min);
  out[idx]   = min(result, digi_max);
}

void CUDA::FITSDigitizerEngine::digitize(const dsp::TimeSeries *input,
    dsp::BitSeries* output, uint64_t ndat, unsigned nbit, float digi_mean, 
    float digi_scale, int digi_min, int digi_max)
{
  if (dsp::Operation::verbose)
    cerr << "CUDA::FITSDigitizerEngine::digitize ndat=" << ndat << " nbit=" 
        << nbit << " mean=" << digi_mean << " digi_scale=" << digi_scale
        << " digi_min=" << digi_min << " digi_max=" << digi_max << endl;

  unsigned ndim = input->get_ndim();
  unsigned nchan = input->get_nchan();
  unsigned npol = input->get_npol();

  // ensure the scratch buffer is large enough
  if (!scratch)
    throw Error (InvalidState, "CUDA::FITSDigitizerEngine::digitize",
                 "scratch space not initialized");

  size_t scratch_needed = ndat * ndim * nchan * npol;
  int * d_scratch = scratch->space<int> (scratch_needed);

  // perform a re-ordering from the input oreder to TFP
  switch (input->get_order())
  {
    case dsp::TimeSeries::OrderTFP:
    {
      // TODO -- this needs to account for reserve
      const float * in = input->get_dattfp();
      unsigned nblocks = ndat;
      unsigned nthreads = 1024;
#ifdef _DEBUG
      cerr << "CUDA::FITSDigitizerEngine::digitize fde_reorder_tfp nblocks=" 
           << nblocks << " nthreads=" << nthreads << endl;
#endif
      fde_reorder_tfp<<<nblocks, nthreads, 0, stream>>>(in, (const unsigned *) mapping, 
                                                        d_offset, d_scale, d_scratch,
                                                        digi_scale, digi_mean,
                                                        nchan, npol);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream( "CUDA::FITSDigitizerEngine::digitize fde_reorder_tfp", stream );
      break;
    }

    case dsp::TimeSeries::OrderFPT:
    {
      const float * first_chanpol = input->get_datptr(0,0);
      uint64_t chanpol_stride = 0;
      if (npol == 1 && nchan > 1)
      {
        const float * next_chanpol = input->get_datptr (1, 0);
        chanpol_stride = next_chanpol - first_chanpol;
      }
      else if (npol > 1)
      {
        const float * next_chanpol = input->get_datptr (0, 1);
        chanpol_stride = next_chanpol - first_chanpol;
      }

      unsigned nthreads = 1024;
      dim3 blocks (ndat / 32, npol, 1);
      if (ndat % 32 != 0)
        blocks.x++;
      unsigned shared_bytes = nthreads * sizeof(int);
#ifdef _DEBUG
      cerr << "CUDA::FITSDigitizerEngine::digitize fde_reorder_fpt blocks=(" 
           << blocks.x << "," << blocks.y << "," << blocks.z << ") " 
           << " nthreads=" << nthreads << " shared_bytes=" << shared_bytes
           << " ndat=" << ndat << " nchan=" << nchan << " npol=" << npol << endl;
#endif
      fde_reorder_fpt<<<blocks, nthreads, shared_bytes, stream>>>(first_chanpol, chanpol_stride, mapping,
                                                       d_offset, d_scale, d_scratch,
                                                       digi_scale, digi_mean, ndat,
                                                       nchan, npol);
      if (dsp::Operation::record_time || dsp::Operation::verbose)
        check_error_stream( "CUDA::FITSDigitizerEngine::digitize fde_reorder_fpt", stream );
      break;
    }

    default:
      throw Error (InvalidState, "CUDA::FITSDigitizerEngine::digitize",
                   "Requires data in TFP or FPT order");
  }

  uint64_t nval = (ndat * nchan * npol * nbit) / 8;
  unsigned nthreads = 1024;
  unsigned nblocks = nval / nthreads;
  if (nval % nthreads != 0)
    nblocks++;

  unsigned char * out = output->get_rawptr();
  // quantise to 8 or 4,2,1
  if (nbit == 8)
  {
#ifdef _DEBUG
    cerr << "CUDA::FITSDigitizerEngine::digitize fde_digitize_8bit nblocks=" 
         << nblocks << " nval=" << nval << " nchan=" << nchan << " npol=" 
         << npol << " digi_min=" << digi_min << " digi_max=" << digi_max 
         << endl;
#endif
    fde_digitize_8bit<<<nblocks, nthreads, 0, stream>>>(d_scratch, out, nval, digi_min, digi_max);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream( "CUDA::FITSDigitizerEngine::digitize fde_digitize_8bit", stream );
  }
  else
  {
#ifdef _DEBUG
    cerr << "CUDA::FITSDigitizerEngine::digitize fde_digitize_nbit nblocks="
         << nblocks << " nval=" << nval << " nbit=" << nbit << " nchan="
         << nchan << " npol=" << npol << " digi_min=" << digi_min 
         << " digi_max=" << digi_max << endl;
#endif
    fde_digitize_nbit<<<nblocks, nthreads, 0, stream>>>(d_scratch, out, nval, nbit, digi_min, digi_max);
    if (dsp::Operation::record_time || dsp::Operation::verbose)
      check_error_stream( "CUDA::FITSDigitizerEngine::digitize fde_digitize_nbit", stream );
  }
}

// copy float scales and offsets from device to host, converting to doubles
void CUDA::FITSDigitizerEngine::get_scale_offsets (double * scale, double * offset, 
                                            unsigned nchan, unsigned npol)
{
#ifdef _DEBUG
  cerr << "CUDA::FITSDigitizerEngine::get_scale_offsets nchan=" << nchan 
       << " npol=" << npol << endl;
#endif

  // copy the scales and offsets from host
  cudaMemcpyAsync (h_offset, d_offset, scale_offset_size, cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync (h_scale, d_scale, scale_offset_size, cudaMemcpyDeviceToHost, stream);
  check_error_stream( "CUDA::FITSDigitizerEngine::get_scale_offsets", stream );

  for (unsigned ichan=0; ichan < nchan; ichan++)
  {
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      unsigned idx = ichan * npol + ipol;
      unsigned odx = ipol * nchan + ichan;
      scale[odx] = double(h_scale[idx]);
      offset[odx] = double(h_offset[idx]);
    }
  }
}
