/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngine.h"

#include "dsp/WeightedTimeSeries.h"
#include "dsp/Response.h"
#include "dsp/Apodization.h"
#include "dsp/InputBuffering.h"
#include "dsp/Scratch.h"
#include "dsp/OptimalFFT.h"

using namespace std;

#define _DEBUG 1

dsp::InverseFilterbank::InverseFilterbank (const char* name, Behaviour behaviour)
  : Convolution (name, behaviour)
{
  set_buffering_policy (new InputBuffering (this));
}

void dsp::InverseFilterbank::prepare ()
{
  if (verbose) {
    cerr << "dsp::InverseFilterbank::prepare" << endl;
  }

  make_preparations ();
  prepared = true;
}

void dsp::InverseFilterbank::reserve ()
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::reserve" << endl;

  resize_output (true);
}

void dsp::InverseFilterbank::set_engine (Engine* _engine)
{
  engine = _engine;
}


void dsp::InverseFilterbank::transformation ()
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::transformation input ndat=" << input->get_ndat()
         << " nchan=" << input->get_nchan() << endl;

  if (!prepared)
    prepare ();

  resize_output ();

  if (has_buffering_policy())
    get_buffering_policy()->set_next_start (input_sample_step * npart);

  uint64_t output_ndat = output->get_ndat();

  // points kept from each small fft
  unsigned nkeep = freq_res - nfilt_tot;

  if (verbose) {
    cerr << "dsp::InverseFilterbank::transformation npart=" << npart
         << " nkeep=" << nkeep << " output_ndat=" << output_ndat << endl;
  }
  // set the input sample
  int64_t input_sample = input->get_input_sample();
  if (output_ndat == 0) {
    output->set_input_sample (0);
  } else if (input_sample >= 0) {
    output->set_input_sample ((input_sample / nsamp_step) * nkeep);
  }

  if (verbose) {
    cerr << "dsp::InverseFilterbank::transformation after prepare output"
            " ndat=" << output->get_ndat() <<
            " input_sample=" << output->get_input_sample() << endl;
  }

  if (!npart)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::transformation empty result" << endl;
    return;
  }

  filterbank ();
}

void dsp::InverseFilterbank::filterbank()
{
  if (engine) {
	  if (verbose) {
		  cerr << "dsp::InverseFilterbank::filterbank: has engine" << endl;
    }
  }

  if (verbose){
    cerr << "dsp::InverseFilterbank::filterbank: computing in_step and out_step" << endl;
  }
  // number of floats to step between input to filterbank
  const uint64_t in_step = input_sample_step * input->get_ndim();
  const uint64_t out_step = output_sample_step * output->get_ndim();

  engine->perform (input, output, npart, in_step, out_step);
  if (Operation::record_time){
    engine->finish ();
  }

}

void dsp::InverseFilterbank::make_preparations ()
{

  // setup the dedispersion discard region for the forward and backward FFTs
  if (has_response()) {
    response = get_response();
    response->match(input, output_nchan);

    output_discard_pos = (int) response->get_impulse_pos();
    output_discard_neg = (int) response->get_impulse_neg();
    output_fft_length = response->get_ndat();
  }

  optimize_discard_region(
    input_discard_neg, input_discard_pos,
    output_discard_neg, output_discard_pos
  );
  optimize_fft_length(input_fft_length, output_fft_length);

  freq_res = output_fft_length;

  input_fft_length *= n_per_sample;
  output_fft_length *= n_per_sample;

  input_discard_total = n_per_sample*(input_discard_neg + input_discard_pos);
  input_sample_step = input_fft_length - input_discard_total;

  output_discard_total = n_per_sample*(output_discard_neg + output_discard_pos);
  output_sample_step = output_fft_length - output_discard_total;

  if (verbose) {
    cerr << "dsp::InverseFilterbankEngineCPU::setup: done optimizing fft lengths and discard regions" << endl;
    cerr << "dsp::InverseFilterbankEngineCPU::setup: input_fft_length="
         << input_fft_length << " output_fft_length="
         << output_fft_length << " input_discard neg/pos="
         << input_discard_neg << "/" << input_discard_pos
         << " output_discard neg/pos"
         << input_discard_neg << "/" << input_discard_pos
         << endl;
  }
  response->set_impulse_neg(output_discard_neg);
  response->set_impulse_pos(output_discard_pos);
  Dedispersion* dedispersion = dynamic_cast<Dedispersion *>(response.ptr());
	if (dedispersion)
	{
		dedispersion->set_frequency_resolution(output_fft_length);
		dedispersion->build();
	}

  if (has_buffering_policy()) {
    if (verbose) {
      cerr << "dsp::InverseFilterbank::make_preparations: reserve="
           << output_fft_length << endl;
    }
    get_buffering_policy()->set_minimum_samples(output_fft_length);
  }

  prepare_output();

  if (engine) {
    engine->setup();
  }

  scalefac = sqrt(engine->get_scalefac());

}

void dsp::InverseFilterbank::prepare_output (uint64_t ndat, bool set_ndat)
{
  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output set ndat=" << ndat << endl;

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

  unsigned tres_ratio = 1;

  if (weighted_output){
    weighted_output->set_reserve_kludge_factor (tres_ratio);
  }

  output->copy_configuration ( get_input() );

  output->set_nchan( nchan );
  output->set_ndim( 2 );
  output->set_state( Signal::Analytic );

  custom_prepare ();

  if (weighted_output)
  {
    weighted_output->set_reserve_kludge_factor (1);
    weighted_output->convolve_weights (input_fft_length, input_sample_step);
    weighted_output->scrunch_weights (tres_ratio);
  }

  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output reset ndat=" << ndat << endl;
    output->resize (ndat);
  }
  else
  {
    ndat = input->get_ndat() / tres_ratio;

    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output scrunch ndat=" << ndat << endl;
    output->resize (ndat);
  }

  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output output ndat="
         << output->get_ndat() << endl;

  output->rescale (scalefac);

  if (verbose) cerr << "dsp::InverseFilterbank::prepare_output scale="
                    << output->get_scale() <<endl;

  /*
   * output data will have new sampling rate
   * NOTE: that nsamp_fft already contains the extra factor of two required
   * when the input TimeSeries is Signal::Nyquist (real) sampled
   */


  double ratechange = (double) input_nchan / (double) output_nchan;
	ratechange /= input->get_oversampling_factor().doubleValue();

 	output->set_rate(input->get_rate() * ratechange);

  if (freq_res == 1){
    output->set_dual_sideband (true);
  }
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
    output->set_swap (true);
  }

  // increment the start time by the number of samples dropped from the fft

  //cerr << "FILTERBANK OFFSET START TIME=" << output_discard_pos << endl;

  output->change_start_time (output_discard_pos);

  if (verbose){
    cerr << "dsp::InverseFilterbank::prepare_output start time += "
         << output_discard_pos << " samps -> " << output->get_start_time() << endl;
  }
  // enable the Response to record its effect on the output Timeseries
  if (response){
    response->mark (output);
  }
}

void dsp::InverseFilterbank::resize_output (bool reserve_extra)
{
  const uint64_t ndat = input->get_ndat();

  // number of big FFTs (not including, but still considering, extra FFTs
  // required to achieve desired time resolution) that can fit into data
  npart = 0;

  if (nsamp_step == 0)
    throw Error (InvalidState, "dsp::InverseFilterbank::resize_output",
                 "nsamp_step == 0 ... not properly prepared");

  if (ndat > input_discard_total)
    npart = (ndat-input_discard_total)/input_sample_step;

  // on some iterations, ndat could be large enough to fit an extra part
  if (reserve_extra && has_buffering_policy()){
    npart += 2;
  }
  uint64_t output_ndat = npart * (output_sample_step/2);

  if (verbose)
    cerr << "dsp::InverseFilterbank::reserve input ndat=" << ndat
         << " overlap=" << nsamp_overlap << " step=" << nsamp_step
         << " reserve=" << reserve_extra << " output_sample_step=" << output_sample_step
         << " npart=" << npart << " output ndat=" << output_ndat << endl;

// #if DEBUGGING_OVERLAP
//   // this exception is useful when debugging, but not at the end-of-file
//   if ( !has_buffering_policy() && ndat > 0
//        && (nsamp_step*npart + nsamp_overlap != ndat) )
//     throw Error (InvalidState, "dsp::InverseFilterbank::reserve",
//                  "npart=%u * step=%u + overlap=%u != ndat=%u",
// 		 npart, nsamp_step, nsamp_overlap, ndat);
// #endif

  // prepare the output TimeSeries
  prepare_output (output_ndat, true);
}

div_t dsp::Filterbank::_calc_lcf (
	int a, int b, Rational os)
{
	return div (a, os.get_denominator()*b);
}

void dsp::Filterbank::_optimize_fft_length (
	int* _input_fft_length, int* _output_fft_length
)
{
	const Rational os = get_oversampling_factor();
	unsigned n_input_channels = input->get_nchan();

	div_t max_input_fft_length_lcf = _calc_lcf(*_output_fft_length, n_input_channels, os);
  while (max_input_fft_length_lcf.rem != 0 || fmod(log2(max_input_fft_length_lcf.quot), 1) != 0){
    if (max_input_fft_length_lcf.rem != 0) {
      (*_output_fft_length) -= max_input_fft_length_lcf.rem;
    } else {
      (*_output_fft_length) -= 2;
    }
    max_input_fft_length_lcf = _calc_lcf(*_output_fft_length, n_input_channels, os);
  }
  *_input_fft_length = max_input_fft_length_lcf.quot * os.get_numerator();
}

void dsp::Filterbank::_optimize_discard_region(
  int* _input_discard_neg,
  int* _input_discard_pos,
  int* _output_discard_neg,
  int* _output_discard_pos,
)
{
	const Rational os = get_oversampling_factor();
	unsigned n_input_channels = input->get_nchan();
	vector<int> n = {_output_discard_pos, _output_discard_neg};
  vector<div_t> lcfs(2);
  int min_n;
	div_t lcf;
  for (int i=0; i<n.size(); i++) {
    min_n = n[i];
    lcf = _calc_lcf(min_n, n_input_channels, os);
    if (lcf.rem != 0) {
      min_n += os.get_denominator()*n_input_channels - lcf.rem;
			lcf.quot += 1;
	    lcf.rem = 0;
    }
    lcfs[i] = lcf;
    n[i] = min_n;
  }
  *_output_discard_pos = n[0];
  *_output_discard_neg = n[1];
  *_input_discard_pos = lcfs[0].quot * os.get_numerator();
  *_input_discard_neg = lcfs[1].quot * os.get_numerator();
}
