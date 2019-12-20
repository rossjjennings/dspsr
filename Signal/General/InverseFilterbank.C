/***************************************************************************
 *
 *   Copyright (C) 2018 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/
#include <vector>

#include "dsp/FIRFilter.h"
#include "dsp/WeightedTimeSeries.h"
#include "dsp/Response.h"
#include "dsp/Dedispersion.h"
#include "dsp/Apodization.h"
#include "dsp/InputBuffering.h"
#include "dsp/Scratch.h"
#include "dsp/OptimalFFT.h"

#include "dsp/InverseFilterbank.h"
#include "dsp/InverseFilterbankEngine.h"


using namespace std;

#define _DEBUG 1

dsp::InverseFilterbank::InverseFilterbank (const char* name, Behaviour behaviour)
  : Convolution (name, behaviour)
{
  set_buffering_policy (new InputBuffering (this));
  input_discard_neg = 0;
  input_discard_pos = 0;
  input_discard_total = 0;
  output_discard_neg = 0;
  output_discard_pos = 0;
  output_discard_total = 0;

  pfb_all_chan = false;
  pfb_dc_chan = false;

}

void dsp::InverseFilterbank::set_input (const dsp::TimeSeries* input)
{
  if (verbose) {
    std::cerr << "dsp::InverseFilterbank::set_input(" << input << ")" << std::endl;
    std::cerr << "dsp::InverseFilterbank::set_input: input->get_oversampling_factor()="
      << input->get_oversampling_factor() << std::endl;
  }
  dsp::Convolution::set_input(input);
  oversampling_factor = input->get_oversampling_factor();
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


  if (has_buffering_policy()) {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbank::transformation: get_buffering_policy()->set_next_start("
        << input_sample_step * npart << ")" << std::endl;
    }
    get_buffering_policy()->set_next_start (input_sample_step * npart);
  }


  uint64_t output_ndat = output->get_ndat();

  int64_t input_sample = input->get_input_sample();
  int64_t new_output_sample = 0;

  if (input_sample >= 0) {
    // unsigned nkeep = freq_res - nfilt_tot;
    // new_output_sample = (input_sample / input_sample_step) * nkeep;
    new_output_sample = output_ndat;
  }

  if (verbose) {
    std::cerr << "dsp::InverseFilterbank::transformation: setting input sample to "
      << new_output_sample << std::endl;
  }

  output->set_input_sample (new_output_sample);
  if (zero_DM) {
    get_zero_DM_output()->set_input_sample(new_output_sample);
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
  if (verbose) {
	  cerr << "dsp::InverseFilterbank::filterbank" << endl;
  }

  if (verbose){
    cerr << "dsp::InverseFilterbank::filterbank: computing in_step and out_step" << endl;
  }
  // number of floats to step between input to filterbank
  const uint64_t in_step = input_sample_step * input->get_ndim();
  const uint64_t out_step = output_sample_step * output->get_ndim();

  engine->perform (input, output, zero_DM_output, npart, in_step, out_step);
  if (Operation::record_time){
    engine->finish ();
  }

}

inline unsigned output_to_input (
  unsigned n,
  unsigned _input_n,
  unsigned _output_n,
  const Rational& osf
)
{
  return (n*osf.get_numerator()) / (osf.get_denominator()*(_input_n/_output_n));
}


void dsp::InverseFilterbank::make_preparations ()
{
  if (verbose) {
    std::cerr << "dsp::InverseFilterbank::make_preparations:"
      << " oversampling_factor=" << get_oversampling_factor()
      << " input->get_oversampling_factor()=" << input->get_oversampling_factor()
      << " pfb_dc_chan=" << get_pfb_dc_chan()
      << " pfb_all_chan=" << get_pfb_all_chan()
      << " fft_window_str=" << get_fft_window_str()
      << std::endl;
  }
  if (oversampling_factor != input->get_oversampling_factor()) {
    oversampling_factor = input->get_oversampling_factor();
  }

  bool real_to_complex = (input->get_state() == Signal::Nyquist);
  unsigned n_per_sample = real_to_complex ? 2: 1;
  Rational osf = get_oversampling_factor();
  // setup the dedispersion discard region for the forward and backward FFTs
  input_nchan = input->get_nchan();
  if (verbose) {
    std::cerr << "dsp::InverseFilterbank::make_preparations: has_response() " << has_response() << std::endl;
  }
  if (has_response()) {
    response->match(input, output_nchan);
    if (zero_DM && has_zero_DM_response()) {
      zero_DM_response->match(input, output_nchan);
    }
    output_discard_pos = response->get_impulse_pos();
    output_discard_neg = response->get_impulse_neg();
    freq_res = response->get_ndat();
    output_fft_length = freq_res;

    input_discard_pos = output_to_input(output_discard_pos, input_nchan, output_nchan, osf);
    input_discard_neg = output_to_input(output_discard_neg, input_nchan, output_nchan, osf);
    input_fft_length = output_to_input(freq_res, input_nchan, output_nchan, osf);

    input_fft_length *= n_per_sample;
    output_fft_length *= n_per_sample;

    input_discard_total = n_per_sample*(input_discard_neg + input_discard_pos);
    input_sample_step = input_fft_length - input_discard_total;

    output_discard_total = n_per_sample*(output_discard_neg + output_discard_pos);
    output_sample_step = output_fft_length - output_discard_total;

  } else {
    output_fft_length = input_nchan*get_oversampling_factor().normalize(input_fft_length) / output_nchan;
    output_fft_length *= n_per_sample;
    input_fft_length *= n_per_sample;
    output_sample_step = output_fft_length;
    input_sample_step = input_fft_length;
  }

  if (verbose) {
    std::cerr << "dsp::InverseFilterbank::make_preparations: output_fft_length="
      << output_fft_length << std::endl;
    std::cerr << "dsp::InverseFilterbank::make_preparations: input_fft_length="
      << input_fft_length << std::endl;
    std::cerr << "dsp::InverseFilterbank::make_preparations: output_sample_step="
      << output_sample_step << std::endl;
    std::cerr << "dsp::InverseFilterbank::make_preparations: input_sample_step="
      << input_sample_step << std::endl;
    std::cerr << "dsp::InverseFilterbank::make_preparations:"
      << " input_discard_neg=" << input_discard_neg
      << " input_discard_pos=" << input_discard_pos
      << std::endl;
    std::cerr << "dsp::InverseFilterbank::make_preparations:"
      << " output_discard_neg=" << output_discard_neg
      << " output_discard_pos=" << output_discard_pos
      << std::endl;

    std::cerr << "dsp::InverseFilterbank::make_preparations: freq_res="
      << freq_res << std::endl;
  }

  if (has_apodization()) {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbank::make_preparations: creating FFT window" << std::endl;
      std::cerr << "dsp::InverseFilterbank::make_preparations: fft_window_str=" << fft_window_str << std::endl;
    }

    dsp::Apodization* fft_window = get_apodization();
    dsp::Apodization::Type fft_window_type = dsp::Apodization::type_map[fft_window_str];
    fft_window->set_shape(
      input_fft_length,
      fft_window_type,
      true,
      input_discard_pos,
      0
    );
  }


  if (verbose) {
    cerr << "dsp::InverseFilterbank::make_preparations: done optimizing fft lengths and discard regions" << endl;
    cerr << "dsp::InverseFilterbank::make_preparations: input_fft_length="
         << input_fft_length << " output_fft_length="
         << output_fft_length << " input_discard_neg/pos="
         << input_discard_neg << "/" << input_discard_pos
         << " output_discard_neg/pos="
         << output_discard_neg << "/" << output_discard_pos
         << endl;
  }

  if (has_buffering_policy()) {
    if (verbose) {
      cerr << "dsp::InverseFilterbank::make_preparations: reserve="
           << output_fft_length << endl;
    }
    get_buffering_policy()->set_minimum_samples(output_fft_length);
  }



  scalefac = 1.0;
  if (FTransform::get_norm() == FTransform::unnormalized) {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbank::make_preparations: unnormalized FFT" << std::endl;
    }
    scalefac = pow(double(output_fft_length), 2);
    scalefac *= pow(oversampling_factor.doubleValue(), 2);
  } else {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbank::make_preparations: normalized FFT" << std::endl;
    }
  }

  prepare_output ();

  if (verbose) {
    cerr << "dsp::InverseFilterbank::make_preparations: "
         << "calling engine setup" << endl;
  }
  engine->setup (this);

  if (verbose) {
    cerr << "dsp::InverseFilterbank::make_preparations:"
         << " scalefac=" << scalefac
         << endl;
  }

  if (verbose) {
    std::cerr << "dsp::InverseFilterbank::make_preparations: allocating "
      << engine->get_total_scratch_needed() << " floats for engine scratch space"
      << std::endl;
  }

  float* scratch_space = scratch->space<float>(engine->get_total_scratch_needed());
  engine->set_scratch(scratch_space);

}

void dsp::InverseFilterbank::prepare_output (uint64_t ndat, bool set_ndat)
{
  if (verbose)
    cerr << "dsp::InverseFilterbank::prepare_output" << endl;

  if (set_ndat)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output set ndat=" << ndat << endl;

    output->set_npol( input->get_npol() );
    output->set_nchan( output_nchan );
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

  output->copy_configuration(get_input());

  output->set_nchan(output_nchan);
  output->set_ndim(2);
  output->set_state(Signal::Analytic);



  if (verbose) {
    cerr << "dsp::InverseFilterbank::prepare_output"
         << " output ndim=" << output->get_ndim()
         << " output npol=" << output->get_npol()
         << " output nchan=" << output->get_nchan()
         << " output ndat=" << output->get_ndat()
         << endl;
  }

  custom_prepare ();

  if (weighted_output)
  {
    if (verbose)
      cerr << "dsp::InverseFilterbank::prepare_output using weighted_output" << endl;
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
                    << output->get_scale() << endl;

  /*
   * output data will have new sampling rate
   * NOTE: that nsamp_fft already contains the extra factor of two required
   * when the input TimeSeries is Signal::Nyquist (real) sampled
   */

  double ratechange = static_cast<double>(input_nchan) / static_cast<double>(output_nchan);
  if (verbose) {
    cerr << "dsp::InverseFilterbank::prepare_output"
         << " ratechange=" << ratechange
         << " input_nchan=" << static_cast<double>(input_nchan)
         << " output_nchan=" << static_cast<double>(output_nchan)
         << endl;
  }
	ratechange /= input->get_oversampling_factor().doubleValue();

 	output->set_rate(input->get_rate() * ratechange);

  if (verbose) {
    cerr << "dsp::InverseFilterbank::prepare_output"
         << " ratechange=" << ratechange
         << " input rate=" << input->get_rate()
         << " output rate=" << output->get_rate()
         << endl;
  }


  if (freq_res == 1){
    output->set_dual_sideband (true);
  }
  /*
   * if freq_res is even, then each sub-band will be centred on a frequency
   * that lies on a spectral bin *edge* - not the centre of the spectral bin
   */

  output->set_dc_centred (freq_res%2);

// #if 0
  // the centre frequency of each sub-band will be offset
  // double channel_bandwidth = input->get_bandwidth() / nchan;
  // double shift = double(freq_res-1)/double(freq_res);
  // output->set_centre_frequency_offset ( 0.5*channel_bandwidth*shift );
// #endif

  // dual sideband data produces a band swapped result
  if (input->get_dual_sideband())
  {
    if (verbose) {
      std::cerr << "dsp::InverseFilterbank::prepare_output: setting output swap to true" << std::endl;
    }
    output->set_swap (true);
  }

  if (verbose) {
    std::cerr << "dsp::InverseFilterbank::prepare_output:"
      << " output->get_dual_sideband()=" << output->get_dual_sideband()
      << " output->get_dc_centred()=" << output->get_dc_centred()
      << std::endl;
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

  if (zero_DM) {
    get_zero_DM_output()->copy_configuration(output);
    get_zero_DM_output()->resize(output->get_ndat());
    get_zero_DM_output()->change_start_time(output_discard_pos);
  }

}

void dsp::InverseFilterbank::resize_output (bool reserve_extra)
{

  if (verbose) {
    cerr << "dsp::InverseFilterbank::resize_output"
         << " reserve_extra=" << reserve_extra
         << endl;
  }
  const uint64_t ndat = input->get_ndat();

  // number of big FFTs (not including, but still considering, extra FFTs
  // required to achieve desired time resolution) that can fit into data
  npart = 0;

  if (input_sample_step == 0) {
    throw Error (InvalidState, "dsp::InverseFilterbank::resize_output",
                 "input_sample_step == 0 ... not properly prepared");
  }

  if (ndat > (unsigned) input_discard_total) {
    npart = (ndat-input_discard_total)/input_sample_step;
  }

  // on some iterations, ndat could be large enough to fit an extra part
  if (reserve_extra && has_buffering_policy()){
    npart += 2;
  }
  uint64_t output_ndat = npart * output_sample_step;
  // uint64_t output_ndat = npart * (output_sample_step/2);

  if (verbose){
    cerr << "dsp::InverseFilterbank::reserve input ndat=" << ndat
         << " overlap=" << input_discard_total << " step=" << input_sample_step
         << " reserve=" << reserve_extra << " output_sample_step=" << output_sample_step
         << " npart=" << npart << " output ndat=" << output_ndat << endl;
   }
// #if DEBUGGING_OVERLAP
//   // this exception is useful when debugging, but not at the end-of-file
//   if ( !has_buffering_policy() && ndat > 0
//        && (input_sample_step*npart + nsamp_overlap != ndat) )
//     throw Error (InvalidState, "dsp::InverseFilterbank::reserve",
//                  "npart=%u * step=%u + overlap=%u != ndat=%u",
// 		 npart, input_sample_step, nsamp_overlap, ndat);
// #endif

  // prepare the output TimeSeries
  prepare_output (output_ndat, true);
}

// div_t dsp::InverseFilterbank::calc_lcf (
// 	int a, int b, Rational os)
// {
// 	return div (a, os.get_denominator()*b);
// }
//
// void dsp::InverseFilterbank::optimize_fft_length (
// 	int* _input_fft_length, int* _output_fft_length
// )
// {
//   if (verbose) {
//     cerr << "dsp::InverseFilterbank::optimize_fft_length" << endl;
//   }
// 	const Rational os = get_oversampling_factor();
// 	int n_input_channels = (int) input->get_nchan();
//
// 	div_t max_input_fft_length_lcf = calc_lcf(*_output_fft_length, n_input_channels, os);
//   while (max_input_fft_length_lcf.rem != 0 || fmod(log2(max_input_fft_length_lcf.quot), 1) != 0){
//     if (max_input_fft_length_lcf.rem != 0) {
//       (*_output_fft_length) -= max_input_fft_length_lcf.rem;
//     } else {
//       (*_output_fft_length) -= 2;
//     }
//     max_input_fft_length_lcf = calc_lcf(*_output_fft_length, n_input_channels, os);
//   }
//   *_input_fft_length = max_input_fft_length_lcf.quot * os.get_numerator();
// }
//
// void dsp::InverseFilterbank::optimize_discard_region(
//   int* _input_discard_neg,
//   int* _input_discard_pos,
//   int* _output_discard_neg,
//   int* _output_discard_pos
// )
// {
//   if (verbose) {
//     cerr << "dsp::InverseFilterbank::optimize_discard_region" << endl;
//   }
// 	const Rational os = get_oversampling_factor();
// 	int n_input_channels = (int) input->get_nchan();
//   // if (verbose) {
//   //   cerr << "dsp::InverseFilterbank::optimize_discard_region"
//   //         << " oversampling_factor=" << os
//   //         << " n_input_channels=" << n_input_channels
//   //         << endl;
//   // }
//
// 	vector<int> n = {*_output_discard_pos, *_output_discard_neg};
//   vector<div_t> lcfs(2);
//   int min_n;
// 	div_t lcf;
//   for (int i=0; i<n.size(); i++) {
//     min_n = n[i];
//     lcf = calc_lcf(min_n, n_input_channels, os);
//     if (lcf.rem != 0) {
//       min_n += os.get_denominator()*n_input_c hannels - lcf.rem;
// 			lcf.quot += 1;
// 	    lcf.rem = 0;
//     }
//     lcfs[i] = lcf;
//     n[i] = min_n;
//   }
//
//   *_output_discard_pos = n[0];
//   *_output_discard_neg = n[1];
//   *_input_discard_pos = lcfs[0].quot * os.get_numerator();
//   *_input_discard_neg = lcfs[1].quot * os.get_numerator();
// }
