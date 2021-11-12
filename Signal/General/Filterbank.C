/***************************************************************************
 *
 *   Copyright (C) 2002-2011 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/Filterbank.h"
#include "dsp/FilterbankEngine.h"

#include "dsp/WeightedTimeSeries.h"
#include "dsp/Response.h"
#include "dsp/Apodization.h"
#include "dsp/InputBuffering.h"
#include "dsp/Scratch.h"
#include "dsp/OptimalFFT.h"

#include "FTransform.h"

#include <fstream>

using namespace std;

// #define _DEBUG 1

dsp::Filterbank::Filterbank (const char* name, Behaviour behaviour)
  : Convolution (name, behaviour)
{
  nchan = 0;
  freq_res = 1;
  // overlap_ratio = 0;

  set_buffering_policy (new InputBuffering (this));
}

void dsp::Filterbank::set_engine (Engine* _engine)
{
  engine = _engine;
}

dsp::Filterbank::Engine* dsp::Filterbank::get_engine ()
{
  return engine;
}

void dsp::Filterbank::prepare ()
{
  if (verbose)
    cerr << "dsp::Filterbank::prepare" << endl;

  make_preparations ();
  prepared = true;
}

FTransform::Plan* dsp::Filterbank::get_forward()
{
  if (!engine)
    throw Error (InvalidState, "dsp::Filterbank::get_forward",
                 "no engine configured");
  return engine->get_forward();
}

FTransform::Plan* dsp::Filterbank::get_backward()
{
  if (!engine)
    throw Error (InvalidState, "dsp::Filterbank::get_backward",
                 "no engine configured");
  return engine->get_backward();
}

/*
  These are preparations that could be performed once at the start of
  the data processing
*/
void dsp::Filterbank::make_preparations ()
{
  if (nchan < input->get_nchan() )
    throw Error (InvalidState, "dsp::Filterbank::make_preparations",
		 "output nchan=%d < input nchan=%d",
		 nchan, input->get_nchan());

  if (nchan % input->get_nchan() != 0)
    throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                 "output nchan=%d not a multiple of input nchan=%d",
                 nchan, input->get_nchan());

  //! Number of channels outputted per input channel
  nchan_subband = nchan / input->get_nchan();

  //! Complex samples dropped from beginning of cyclical convolution result
  nfilt_pos = 0;

  //! Complex samples dropped from end of cyclical convolution result
  nfilt_neg = 0;

  if (response)
  {
    if (verbose)
      cerr << "dsp::Filterbank call Response::match" << endl;

    // convolve the data with a frequency response function during
    // filterbank construction...

    response -> match (input, nchan);
    if (response->get_nchan() != nchan)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                   "response nchan=%d != output nchan=%d",
                   response->get_nchan(), nchan);

    nfilt_pos = response->get_impulse_pos ();
    nfilt_neg = response->get_impulse_neg ();

    freq_res = response->get_ndat();

    if (verbose)
      cerr << "dsp::Filterbank Response nfilt_pos=" << nfilt_pos
           << " nfilt_neg=" << nfilt_neg
           << " freq_res=" << response->get_ndat()
           << " ndim=" << response->get_ndim() << endl;

    if (freq_res == 0)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                   "Response.ndat = 0");
  }
  else
  {
    // configure normalizer with the width of the frequency response
    normalizer->set_ndat (freq_res);
    normalizer->match (input, nchan);
  }

  // number of complex values in the result of the first fft
  unsigned n_fft = nchan_subband * freq_res;

  scalefac = 1.0;

  if (verbose)
  {
    string norm = "unknown";
    if (FTransform::get_norm() == FTransform::unnormalized)
      norm = "unnormalized";
    else if (FTransform::get_norm() == FTransform::normalized)
      norm = "normalized";

    cerr << "dsp::Filterbank::make_preparations n_fft=" << n_fft
         << " freq_res=" << freq_res << " fft::norm=" << norm
         << " nchan_subband=" << nchan_subband << endl;
  }

  // number of complex samples invalid in result of small ffts
  nfilt_tot = nfilt_pos + nfilt_neg;

  // number of time samples by which big ffts overlap
  nsamp_overlap = 0;

  // number of time samples in first fft
  nsamp_fft = 0;

  if (input->get_state() == Signal::Nyquist)
  {
    // cerr << "Filterbank::make_preparations real-valued input data" << endl;
    nsamp_fft = 2 * n_fft;
    nsamp_overlap = 2 * nfilt_tot * nchan_subband;
  }
  else if (input->get_state() == Signal::Analytic)
  {
    nsamp_fft = n_fft;
    nsamp_overlap = nfilt_tot * nchan_subband;
  }
  else
    throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                 "invalid input data state = " + tostring(input->get_state()));

  if (FTransform::get_norm() == FTransform::unnormalized)
    scalefac = double(nsamp_fft) * double(freq_res);

  else if (FTransform::get_norm() == FTransform::normalized)
    scalefac = double(nsamp_fft) / double(freq_res);

  // sqrt since the scale factor is applied prior to detection
  scalefac = sqrt(scalefac);

  // configure the normalizing response to ensure FFT lengths do
  // not rescale the data exceedingly
  normalizer->set_scale_factor (1.0 / scalefac);
  scalefac = 1.0;

  if (response)
  {
    if (verbose)
      cerr << "dsp::Filterbank::make_preparations building a response product" << endl;
    response_product = new ResponseProduct ();
    response_product->add_response (response);
    response_product->add_response (normalizer);
    response_product->set_copy_index (0);
    response_product->set_match_index (0);
    response = response_product;
  }
  else
  {
    if (verbose)
      cerr << "dsp::Filterbank::make_preparations using normalized response" << endl;
    response = normalizer;
  }

  response -> match (input, nchan);

  if (zero_DM)
  {
    // configure normalizer with the width of the frequency response
    zero_DM_normalizer = new ScalarFilter();
    zero_DM_normalizer->set_ndat (freq_res);
    zero_DM_normalizer->match (input, nchan);
    zero_DM_normalizer->set_scale_factor (1.0 / scalefac);

    if (zero_DM_response)
    {
      zero_dm_response_product = new ResponseProduct ();
      zero_dm_response_product->add_response (zero_DM_response);
      zero_dm_response_product->add_response (zero_DM_normalizer);
      zero_dm_response_product->set_copy_index (0);
      zero_dm_response_product->set_match_index (0);
      zero_DM_response = zero_dm_response_product;
    }
    else
    {
      cerr << "Using zero_DM_normalizer as zero_DM_response" << endl;
      zero_DM_response = zero_DM_normalizer;
    }
  }

  if (zero_DM_response)
    zero_DM_response->match (input, nchan);

  // number of timesamples between start of each big fft
  nsamp_step = nsamp_fft - nsamp_overlap;

  if (verbose)
    cerr << "dsp::Filterbank::make_preparations nfilt_tot=" << nfilt_tot
         << " nsamp_fft=" << nsamp_fft << " nsamp_step=" << nsamp_step
         << " nsamp_overlap=" << nsamp_overlap << endl;

  // if given, test the validity of the window function
  if (apodization)
  {
    if( input->get_nchan() > 1 )
      throw Error(InvalidState,"dsp::Filterbank::make_preparations",
                  "not implemented for nchan=%d > 1",
                  input->get_nchan());

    if (apodization->get_ndat() != nsamp_fft)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                   "invalid apodization function ndat=%d"
                   " (nfft=%d)", apodization->get_ndat(), nsamp_fft);

    if (input->get_state() == Signal::Analytic
        && apodization->get_ndim() != 2)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                   "Signal::Analytic signal. Real apodization function.");

    if (input->get_state() == Signal::Nyquist
        && apodization->get_ndim() != 1)
      throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                   "Signal::Nyquist signal. Complex apodization function.");
  }

  // matrix convolution
  matrix_convolution = false;

  if (response)
  {
    // if the response has 8 dimensions, then perform matrix convolution
    matrix_convolution = (response->get_ndim() == 8);

    if (verbose)
      cerr << "dsp::Filterbank::make_preparations with "
           << ((matrix_convolution)?"matrix":"complex") << " convolution"
           << endl;

#if 0
    if (matrix_convolution && input->get_nchan() > 1)
      throw Error(InvalidState,"dsp::Filterbank::make_preparations",
                  "matrix convolution untested for > one input channel");
#endif

    if (matrix_convolution && input->get_npol() != 2)
        throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                     "matrix convolution and input.npol != 2");
  }

  if (has_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::Filterbank::make_preparations"
        " reserve=" << nsamp_fft << endl;

    get_buffering_policy()->set_minimum_samples (nsamp_fft);
  }

  prepare_output ();

  // the engine should delete the passband if it doesn't support this feature
  if (passband)
  {
    if (response)
      passband -> match (response);

    unsigned passband_npol = input->get_npol();
    if (matrix_convolution)
      passband_npol = 4;

    passband->resize (passband_npol, input->get_nchan(), n_fft, 1);

    if (!response)
      passband->match (input);
  }

  if (engine)
  {
    if (verbose)
      cerr << "dsp::Filterbank::make_preparations setup engine" << endl;
    engine->setup (this);
    return;
  }
  else
  {
    throw Error (InvalidState, "dsp::Filterbank::make_preparations",
                 "no engine configured");
  }
}

void dsp::Filterbank::prepare_output (uint64_t ndat, bool set_ndat)
{
  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::Filterbank::prepare_output set ndat=" << ndat << endl;

    output->set_npol( input->get_npol() );
    output->set_nchan( nchan );
    output->set_ndim( 2 );
    output->set_state( Signal::Analytic);
    output->resize( ndat );
  }

  WeightedTimeSeries* weighted_output;
  weighted_output = dynamic_cast<WeightedTimeSeries*> (output.get());

  /* the problem: copy_configuration copies the weights array, which
     results in a call to resize_weights, which sets some offsets
     according to the reserve (for later prepend).  However, the
     offset is computed based on values that are about to be changed.
     This kludge allows the offsets to reflect the correct values
     that will be set later */

  unsigned tres_ratio = nsamp_fft / freq_res;

  if (weighted_output)
    weighted_output->set_reserve_kludge_factor (tres_ratio);

  output->copy_configuration ( get_input() );

  output->set_nchan( nchan );
  output->set_ndim( 2 );
  output->set_state( Signal::Analytic );

  custom_prepare ();

  if (weighted_output)
  {
    weighted_output->set_reserve_kludge_factor (1);
    weighted_output->convolve_weights (nsamp_fft, nsamp_step);
    weighted_output->scrunch_weights (tres_ratio);
  }

  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::Filterbank::prepare_output reset ndat=" << ndat << endl;
    output->resize (ndat);
  }
  else
  {
    ndat = input->get_ndat() / tres_ratio;

    if (verbose)
      cerr << "dsp::Filterbank::prepare_output scrunch ndat=" << ndat << endl;
    output->resize (ndat);
  }

  if (verbose)
    cerr << "dsp::Filterbank::prepare_output output ndat="
         << output->get_ndat() << endl;

  output->rescale (scalefac);

  if (verbose) cerr << "dsp::Filterbank::prepare_output scale="
                    << output->get_scale() <<endl;

  /*
   * output data will have new sampling rate
   * NOTE: that nsamp_fft already contains the extra factor of two required
   * when the input TimeSeries is Signal::Nyquist (real) sampled
   */
  double ratechange = double(freq_res) / double (nsamp_fft);
  output->set_rate (input->get_rate() * ratechange);

  if (freq_res == 1)
    output->set_dual_sideband (true);

  /*
   * if freq_res is even, then each sub-band will be centred on a frequency
   * that lies on a spectral bin *edge* - not the centre of the spectral bin
   */
  output->set_dc_centred (freq_res%2);

#if 0
  // the centre frequency of each sub-band will be offset
  double channel_bandwidth = input->get_bandwidth() / nchan;
  double shift = double(freq_res-1)/double(freq_res);
  output->set_centre_frequency_offset ( 0.5*channel_bandwidth*shift );
#endif

  // dual sideband data produces a band swapped result
  if (input->get_dual_sideband())
  {
    if (input->get_nchan() > 1)
      output->set_nsub_swap (input->get_nchan());
    else
      output->set_swap (true);
  }

  // increment the start time by the number of samples dropped from the fft

  //cerr << "FILTERBANK OFFSET START TIME=" << nfilt_pos << endl;

  output->change_start_time (nfilt_pos);

  // zero DM output
  if (zero_DM)
  {
    if (verbose)
      cerr << "dsp::Filterbank::prepare_output copying output configuration to zero_DM_output" << endl;
    zero_DM_output->copy_configuration(output);
    zero_DM_output->resize(output->get_ndat());
  }

  if (verbose)
    cerr << "dsp::Filterbank::prepare_output start time += "
         << nfilt_pos << " samps -> " << output->get_start_time() << endl;

  // enable the Response to record its effect on the output Timeseries
  if (response)
    response->mark (output);
}

void dsp::Filterbank::reserve ()
{
  if (verbose)
    cerr << "dsp::Filterbank::reserve" << endl;

  resize_output (true);
}

void dsp::Filterbank::resize_output (bool reserve_extra)
{
  const uint64_t ndat = input->get_ndat();

  // number of big FFTs (not including, but still considering, extra FFTs
  // required to achieve desired time resolution) that can fit into data
  npart = 0;

  if (nsamp_step == 0)
    throw Error (InvalidState, "dsp::Filterbank::resize_output",
                 "nsamp_step == 0 ... not properly prepared");

  if (ndat > nsamp_overlap)
    npart = (ndat-nsamp_overlap)/nsamp_step;

  // on some iterations, ndat could be large enough to fit an extra part
  if (reserve_extra && has_buffering_policy())
    npart += 2;

  // points kept from each small fft
  unsigned nkeep = freq_res - nfilt_tot;

  uint64_t output_ndat = npart * nkeep;

  if (verbose)
    cerr << "dsp::Filterbank::reserve input ndat=" << ndat
         << " overlap=" << nsamp_overlap << " step=" << nsamp_step
         << " reserve=" << reserve_extra << " nkeep=" << nkeep
         << " npart=" << npart << " output ndat=" << output_ndat << endl;

#if DEBUGGING_OVERLAP
  // this exception is useful when debugging, but not at the end-of-file
  if ( !has_buffering_policy() && ndat > 0
       && (nsamp_step*npart + nsamp_overlap != ndat) )
    throw Error (InvalidState, "dsp::Filterbank::reserve",
                 "npart=%u * step=%u + overlap=%u != ndat=%u",
		 npart, nsamp_step, nsamp_overlap, ndat);
#endif

  // prepare the output TimeSeries
  prepare_output (output_ndat, true);
}

void dsp::Filterbank::transformation ()
{
  if (verbose)
    cerr << "dsp::Filterbank::transformation input ndat=" << input->get_ndat()
         << " nchan=" << input->get_nchan() << endl;

  if (!prepared)
    prepare ();

  resize_output ();

  if (has_buffering_policy())
    get_buffering_policy()->set_next_start (nsamp_step * npart);

  uint64_t output_ndat = output->get_ndat();

  // points kept from each small fft
  unsigned nkeep = freq_res - nfilt_tot;

  if (verbose) {
    cerr << "dsp::Filterbank::transformation npart=" << npart
         << " nkeep=" << nkeep << " output_ndat=" << output_ndat << endl;
  }
  // set the input sample
  int64_t input_sample = input->get_input_sample();
  if (output_ndat == 0) {
    output->set_input_sample (0);
  } else if (input_sample >= 0) {
    output->set_input_sample ((input_sample / nsamp_step) * nkeep);
  }

  if (zero_DM) {
    zero_DM_output->set_input_sample(output->get_input_sample());
  }

  if (verbose) {
    cerr << "dsp::Filterbank::transformation after prepare output"
            " ndat=" << output->get_ndat() <<
            " input_sample=" << output->get_input_sample() << endl;
  }

  if (!npart)
  {
    if (verbose)
      cerr << "dsp::Filterbank::transformation empty result" << endl;
    return;
  }

  filterbank ();
}

void dsp::Filterbank::filterbank ()
{

  // divide up the scratch space
  scratch_needed = engine->get_total_scratch_needed();

  if (verbose) {
    std::cerr << "dsp::Filterbank:filterbank: allocating "<< scratch_needed <<" bytes of scratch space" << std::endl;
  }

  float* scratch_space = scratch->space<float>(scratch_needed);

  if (verbose){
    cerr << "dsp::Filterbank::filterbank: computing in_step and out_step" << endl;
  }
  // number of floats to step between input to filterbank
  const uint64_t in_step = nsamp_step * input->get_ndim();

  // points kept from each small fft
  const unsigned nkeep = freq_res - nfilt_tot;

  // number of floats to step between output from filterbank
  const uint64_t out_step = nkeep * 2;

  engine->set_scratch (scratch_space);
  if (zero_DM)
  {
    if (verbose)
      cerr << "dsp::Filterbank::filterbank engine->perform [ZeroDM]" << endl;
    engine->perform (input, output, zero_DM_output, npart, in_step, out_step);
  }
  else
  {
    if (verbose)
      cerr << "dsp::Filterbank::filterbank engine->perform" << endl;
    engine->perform (input, output, npart, in_step, out_step);
  }

  if (Operation::record_time){
    engine->finish ();
  }

  if (verbose)
    cerr << "dsp::Filterbank::transformation return with output ndat="
         << output->get_ndat() << endl;
}
