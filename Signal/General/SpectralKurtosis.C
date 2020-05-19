/***************************************************************************
 *
 *   Copyright (C) 2020 by Andrew Jameson and Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/SpectralKurtosis.h"
#include "dsp/InputBuffering.h"
#include "dsp/SKLimits.h"

#if HAVE_YAMLCPP
#include <yaml-cpp/yaml.h>
#endif

#include <errno.h>
#include <assert.h>
#include <string.h>

#include <fstream>
#include <algorithm>

using namespace std;

dsp::SpectralKurtosis::SpectralKurtosis()
 : Transformation<TimeSeries,TimeSeries>("SpectralKurtosis", outofplace)
{
  resolution.resize(1);
  resolution[0].set_M( 128 );
  resolution[0].noverlap = 1;

  debugd = 1;

  sums = new TimeSeries;
  estimates_tscr = new TimeSeries;
  zapmask = new BitSeries;

  // SK Detector
  npart_total = 0;
  thresholds_tscr_m.resize(1);
  thresholds_tscr_upper.resize(1);
  thresholds_tscr_lower.resize(1);
  zap_counts.resize(4);
  detection_flags.resize(3);
  std::fill (detection_flags.begin(), detection_flags.end(), false);
  detection_flags.resize(3);
  M_tscr = 0;

  unfiltered_hits = 0;

  prepared = false;

  set_buffering_policy(new InputBuffering(this));
  set_zero_DM_buffering_policy(new InputBuffering(&zero_DM_input_container));

  zero_DM = false;
  set_zero_DM_input(new dsp::TimeSeries);

}

dsp::SpectralKurtosis::~SpectralKurtosis ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::SpectralKurtosis~" << endl;

  float percent_all = 0;
  float percent_skfb = 0;
  float percent_tscr = 0;
  float percent_fscr = 0;

  if (npart_total)
  {
    percent_all  = (100 * (float) zap_counts[ZAP_ALL]  / (float) npart_total);
    percent_skfb = (100 * (float) zap_counts[ZAP_SKFB] / (float) npart_total);
    percent_tscr = (100 * (float) zap_counts[ZAP_TSCR] / (float) npart_total);
    percent_fscr = (100 * (float) zap_counts[ZAP_FSCR] / (float) npart_total);
  }

  cerr << "Zapped: "
       << " total=" << percent_all <<  "\%" << " skfb=" << percent_skfb << "\%"
       << " tscr=" << percent_tscr << "\%" << " fscr=" << percent_fscr << "\%"
       << endl;

  delete sums;
  delete estimates_tscr;
  delete zapmask;
}

bool dsp::SpectralKurtosis::get_order_supported (TimeSeries::Order order) const
{
  if (order == TimeSeries::OrderFPT || order == TimeSeries::OrderTFP)
    return true;
  return false;
}


void dsp::SpectralKurtosis::set_engine (Engine* _engine)
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::set_engine()" << endl;
  engine = _engine;
}

#if HAVE_YAMLCPP
void parse_ranges (YAML::Node node,
		   vector< pair<unsigned,unsigned> >& ranges)
{
  if (!node.IsSequence())
    throw Error (InvalidState, "parse_ranges",
		 "YAML::Node is not a sequence");

  if (node[0].IsSequence())
  {
    for (unsigned i=0; i < node.size(); i++)
      parse_ranges (node[i], ranges);
    return;
  }

  std::pair<unsigned,unsigned> range;
  range = node.as< std::pair<unsigned,unsigned> >();
  ranges.push_back(range);
}
#endif

//! Load configuration from YAML filename
void dsp::SpectralKurtosis::load_configuration (const std::string& filename)
{
#if HAVE_YAMLCPP
  YAML::Node node = YAML::LoadFile(filename);

  unsigned nres = 1;

  // different resolutions are specified in a sequence
  if (node.IsSequence())
    nres = node.size();

#if _DEBUG
  cerr << "dsp::SpectralKurtosis::load_configuration " << filename << endl;
  cerr << "dsp::SpectralKurtosis::load_configuration nodes=" << nres << endl;
#endif

  resolution.resize( nres );

  for (unsigned ires=0; ires<nres; ires++)
  {
    YAML::Node one;
    if (node.IsSequence())
      one = node[ires];
    else
      one = node;

    if ( !one["M"] )
      throw Error (InvalidState, "SpectralKurtosis::load_configuration",
		   "M not specified for resolution[%u]", ires);

    resolution[ires].set_M( one["M"].as<unsigned>() );

    // the rest are optional
    
    if ( one["overlap"] )
      resolution[ires].noverlap = one["overlap"].as<unsigned>();

    if ( one["exclude"] )
      parse_ranges( one["exclude"], resolution[ires].exclude );

    if ( one["include"] )
      parse_ranges( one["include"], resolution[ires].include );
  }

#else
  throw Error (InvalidState, "dsp::SpectralKurtosis::load_configuration",
               "not implemented - requires yaml-cpp");
#endif 
}

void dsp::SpectralKurtosis::set_zero_DM_input (TimeSeries* _zero_DM_input)
{
  zero_DM_input_container.set_input(_zero_DM_input);
}

bool dsp::SpectralKurtosis::has_zero_DM_input () const {
  return zero_DM_input_container.has_input();
}

const dsp::TimeSeries* dsp::SpectralKurtosis::get_zero_DM_input () const {
  return zero_DM_input_container.get_input();
}

dsp::TimeSeries* dsp::SpectralKurtosis::get_zero_DM_input () {
  return const_cast<dsp::TimeSeries*>(zero_DM_input_container.get_input());
}

void dsp::SpectralKurtosis::set_M (const std::vector<unsigned>& M)
{
  // cerr << "SpectralKurtosis::set_M size=" << M.size() << endl;

  resize_resolution (M.size());
  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    if (M.size() == 1)
      resolution[ires].set_M (M[0]);
    else
    {
      // cerr << "SpectralKurtosis::set_M[" << ires << "]=" << M[ires] << endl;
      resolution[ires].set_M (M[ires]);
    }
  }
}

void dsp::SpectralKurtosis::set_noverlap (const std::vector<unsigned>& noverlap)
{ 
  // cerr << "SpectralKurtosis::set_noverlap size=" << noverlap.size() << endl;

  resize_resolution (noverlap.size());
  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    if (noverlap.size() == 1)
      resolution[ires].noverlap = noverlap[0];
    else
    {
      // cerr << "SpectralKurtosis::set_noverlap[" << ires << "]=" << noverlap[ires] << endl;
      resolution[ires].noverlap = noverlap[ires];
    }
  }
}

void dsp::SpectralKurtosis::set_thresholds (const std::vector<float>& std_devs)
{
  resize_resolution (std_devs.size());
  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    unsigned jres = (std_devs.size() == 1) ? 0 : ires;
    resolution[ires].set_std_devs( std_devs[jres] );
  }
}

void dsp::SpectralKurtosis::resize_resolution (unsigned nres)
{
  if (nres == 1)
    return;

  if (resolution.size() == 1)
    resolution.resize (nres);
  else if (nres != resolution.size())
    throw Error (InvalidParam, "dsp::SpectralKurtosis::resize_resolution",
                 "size mismatch n=%u != size=%u", nres, resolution.size());
}

bool dsp::SpectralKurtosis::by_M (const Resolution& A, const Resolution& B)
{
  return A.get_M() < B.get_M();
}

void dsp::SpectralKurtosis::Resolution::set_M (unsigned _M)
{
  thresholds.resize(0);
  M=_M;
}

void dsp::SpectralKurtosis::Resolution::set_std_devs (float _std_devs)
{
  thresholds.resize(0);
  std_devs=_std_devs;
}

const std::vector<float>&
dsp::SpectralKurtosis::Resolution::get_thresholds () const
{
  if (thresholds.size() == 0)
    set_thresholds (std_devs);

  return thresholds;
}

void dsp::SpectralKurtosis::Resolution::prepare (uint64_t ndat)
{
  if (M % noverlap)
    throw Error (InvalidState, "dsp::SpectralKurtosis::Resolution::prepare",
                 "noverlap=%u does not divide M=%u", noverlap, M);

  // overlap_offset = idat_off in SpectralKurtosisInputBuffering.pptx
  overlap_offset = M / noverlap;

  if (ndat < M)
  {
    npart = 0;
    output_ndat = 0;
  }
  else
  {
    // npart = N_block in SpectralKurtosisInputBuffering.pptx
    npart = (ndat-M) / overlap_offset + 1;

    /* with reference to diagram in SpectralKurtosisInputBuffering.pptx
       where idat_off = overlap_offset     :NZAPP-208: WvS - 2020-05-19 */

    uint64_t idat_last = M + (npart-1) * overlap_offset;
    uint64_t idat_end = idat_last - overlap_offset;
    unsigned idat_start = M - overlap_offset;

    output_ndat = idat_end - idat_start;
  }
}

void dsp::SpectralKurtosis::Resolution::compatible (Resolution& smaller)
{
  if (M % smaller.M)
    throw Error (InvalidState, "dsp::SpectralKurtosis::Resolution::compatible",
                 "this.M=%u not divisible by that.M=%u", M, smaller.M);

  if (overlap_offset % smaller.M)
    throw Error (InvalidState, "dsp::SpectralKurtosis::Resolution::compatible",
                 "overlap_offset=%u not divisible by that.M=%u", 
                 overlap_offset, smaller.M);
}


void dsp::SpectralKurtosis::Resolution::add_include
(const std::pair<unsigned, unsigned>& range)
{
  include.push_back(range);
}

void dsp::SpectralKurtosis::Resolution::add_exclude
(const std::pair<unsigned, unsigned>& range)
{
  exclude.push_back(range);
}

void set_mask (vector<bool>& mask,
	       const vector< pair<unsigned,unsigned> >& ranges, bool value)
{
  for (unsigned irange=0; irange<ranges.size(); irange++)
  {
    unsigned first = ranges[irange].first;
    unsigned last = ranges[irange].second;

    assert (last < mask.size());
    
    for (unsigned imask=first; imask <= last; imask++)
      mask[imask] = value;
  }
}
    
//! Get the channels to be zapped
const std::vector<bool>&
dsp::SpectralKurtosis::Resolution::get_channels (unsigned nchan) const
{
  if (nchan == channels.size())
    return channels;

  channels.resize (nchan);

  if (include.size() == 0)
    std::fill (channels.begin(), channels.end(), true);
  else
    std::fill (channels.begin(), channels.end(), false);

  set_mask (channels, include, true);
  set_mask (channels, exclude, false);
  
  return channels;
}

/*
 * These are preparations that could be performed once at the start of
 * the data processing
 */
void dsp::SpectralKurtosis::prepare ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::prepare()" << endl;

  std::sort (resolution.begin(), resolution.end(), by_M);

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    resolution[ires].prepare();
    if (ires > 0)
      resolution[ires].compatible( resolution[ires-1] );
  }

  nchan = input->get_nchan();
  npol = input->get_npol();
  ndim = input->get_ndim();

  Memory * memory = const_cast<Memory *>(input->get_memory());
  sums->set_memory (memory);
  estimates_tscr->set_memory (memory);
  zapmask->set_memory (memory);

  // resolution vector sorted by M
  unsigned max_M = resolution.back().get_M();

  if (has_buffering_policy())
  {
    get_buffering_policy()->set_minimum_samples (max_M);
    if (zero_DM)
      get_zero_DM_buffering_policy()->set_minimum_samples (max_M);
  }

  if (engine)
  {
    engine->setup ();
  }
  else
  {
    if (!detection_flags[1])
    {
      S1_tscr.resize(nchan * npol);
      S2_tscr.resize(nchan * npol);
    }
  }

  // ensure output containers are configured correctly
  prepare_output ();

  prepared = true;
}

/*! ensure output parameters are configured correctly */
void dsp::SpectralKurtosis::prepare_output ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::prepare_output()" << endl;

  // Resolution::compatible enforces decreasing overlap_offset order
  unsigned min_offset = resolution.front().overlap_offset;
  unsigned max_npart = resolution.front().npart;

  double mask_rate = input->get_rate() / min_offset;

  sums->copy_configuration (get_input());
  sums->set_order (TimeSeries::OrderTFP);  // stored in TFP order
  sums->set_scale (1.0);                   // no scaling
  sums->set_rate (mask_rate);              // rate is *= noverlap/M

  if (input->get_npol() == 2)
    sums->set_state (Signal::PPQQ);
  else
    sums->set_state (Signal::Intensity);

  sums->set_ndim (2);                      // S1_sum and S2_sum

  double tscrunch_mask_rate = mask_rate;

  if (max_npart > 0)
    tscrunch_mask_rate /= max_npart;

  // tscrunched estimates have same configuration as sums with following changes
  estimates_tscr->copy_configuration (sums);
  estimates_tscr->set_order (TimeSeries::OrderTFP);  // stored in TFP order
  estimates_tscr->set_rate (tscrunch_mask_rate);
  estimates_tscr->set_ndim (1);

  // zap mask has same configuration as sums with following changes
  zapmask->copy_configuration (sums);
  zapmask->set_nbit (8);
  zapmask->set_npol (1);
  zapmask->set_ndim (1);

  // configure output timeseries (out-of-place) to match input
  output->copy_configuration (get_input());

  /* NZAPP-208 - see SpectralKurtosisInputBuffering.pptx */
  unsigned idat_off = resolution.back().overlap_offset;
  unsigned idat_start = resolution.back().get_M() - idat_off;

  output->set_input_sample (input->get_input_sample () + idat_start);
  output->change_start_time (idat_start);

  if (zero_DM)
  {
    if (get_zero_DM_input()->get_input_sample() != input->get_input_sample ())
      throw Error (InvalidState, "dsp::SpectralKurtosis::prepare_output", 
                   "mismatch between normal and zero_DM input samples");
  }

  // if (zero_DM) {
  //   get_zero_DM_input()->set_input_sample(input->get_input_sample());
  // }
}

/* ensure containers have correct dynamic size */
void dsp::SpectralKurtosis::reserve ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::reserve()" << endl;

  const uint64_t ndat  = input->get_ndat();

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    resolution[ires].prepare (ndat);
    if (ires > 0)
      resolution[ires].compatible( resolution[ires-1] );
  }

  unsigned max_npart = resolution.front().npart;
  unsigned min_output_ndat = resolution.back().output_ndat;
  unsigned max_M = resolution.back().get_M();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::reserve input_ndat=" << ndat
         << " max npart=" << max_npart 
         << " output_ndat=" << min_output_ndat << endl;

  // use resize since out of place operation
  sums->resize (max_npart);
  estimates_tscr->resize (max_npart > 0); // 1 if npart != 0
  zapmask->resize (max_npart);

  // reserve space to hold one more than current
  output->resize (min_output_ndat);
}

/* call set of transformations */
void dsp::SpectralKurtosis::transformation ()
{
  if (zero_DM && has_zero_DM_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation"
              " zero_DM_buffering_policy pre_transformation" << endl;

    get_zero_DM_buffering_policy()->pre_transformation();
  }

  if (!prepared)
    prepare();

  const uint64_t ndat  = input->get_ndat();
  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input ndat=" << ndat << endl;

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    resolution[ires].prepare (ndat);
    if (ires > 0)
      resolution[ires].compatible( resolution[ires-1] );
  }

  unsigned max_npart = resolution.front().npart;
  unsigned min_output_ndat = resolution.back().output_ndat;

  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::transformation input max"
         << " npart=" << max_npart
         << " output_ndat=" << min_output_ndat << endl;

  if (has_buffering_policy())
  {
    if (verbose || debugd < 1)
      cerr << "dsp::SpectralKurtosis::transformation setting next_start_sample="
           << min_output_ndat << endl;

    get_buffering_policy()->set_next_start (min_output_ndat);

    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation set_next_start done" << endl;
  }

  if (zero_DM && has_zero_DM_buffering_policy())
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation zero_DM_buffering_policy set_next_start " << endl;
 
    get_zero_DM_buffering_policy()->set_next_start(min_output_ndat);
  }

  prepare_output ();

  // ensure output containers are sized correctly
  reserve();

  if ((ndat == 0) || (max_npart == 0))
    return;

  // perform SK functions
  if (verbose)
  {
    cerr << "dsp::SpectralKurtosis::transformation: calling compute" << endl;
    cerr << "dsp::SpectralKurtosis::transformation:: detection_flags=["
      << detection_flags[0] << ", "
      << detection_flags[1] << ", "
      << detection_flags[2] << "]" << endl;
  }

  compute ();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::transformation: calling detect" << endl;

  if (report)
  {
    float_reporter.emit(
      "input",
      (float*) input->get_datptr(),
      input->get_nchan(),
      input->get_npol(),
      input->get_ndat(),
      input->get_ndim());

    float_reporter.emit(
      "sums",
      sums->get_dattfp(),
      sums->get_nchan(),
      sums->get_npol(),
      sums->get_ndat(),
      sums->get_ndim());

    float_reporter.emit(
      "estimates_tscr",
      estimates_tscr->get_dattfp(),
      estimates_tscr->get_nchan(),
      estimates_tscr->get_npol(),
      estimates_tscr->get_ndat(),
      estimates_tscr->get_ndim());
  }

  detect ();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::transformation: calling mask" << endl;

  if (report)
  {
    char_reporter.emit(
      "zapmask",
      zapmask->get_datptr(),
      zapmask->get_nchan(),
      zapmask->get_npol(),
      zapmask->get_ndat(),
      zapmask->get_ndim());
  }

  mask ();

  if (verbose)
    cerr << "dsp::SpectralKurtosis::transformation: done" << endl;

  if (report)
  {
    float_reporter.emit(
      "output",
      output->get_datptr(),
      output->get_nchan(),
      output->get_npol(),
      output->get_ndat(),
      output->get_ndim());
  }

  //insertsk();
}

void dsp::SpectralKurtosis::compute ()
{
  if (verbose) {
    cerr << "dsp::SpectralKurtosis::compute" << endl;
  }
  const dsp::TimeSeries* compute_input = get_input();

  if (zero_DM)
  {
    compute_input = get_zero_DM_input();
    if (verbose)
    {
      cerr << "dsp::SpectralKurtosis::compute: using zero DM input" << endl;
      cerr << "dsp::SpectralKurtosis::compute: sample input=" << input->get_input_sample() << " zero_DM_input=" << get_zero_DM_input()->get_input_sample() << endl;
      cerr << "dsp::SpectralKurtosis::compute: ndate input=" << input->get_ndat() <<  " zero_DM_input=" << get_zero_DM_input()->get_ndat() << endl;
    }
    // compute_input->set_input_sample(input->get_input_sample());
  }

  unsigned M = resolution.front().get_M();
  unsigned noverlap = resolution.front().noverlap;
  unsigned npart = resolution.front().npart;
  unsigned overlap_offset = resolution.front().overlap_offset;

  if (engine)
  {
    engine->compute (compute_input, sums, estimates_tscr, M);
  }
  else
  {
    // initialise tscr
    if (!detection_flags[1])
    {
      std::fill(S1_tscr.begin(), S1_tscr.end(), 0);
      std::fill(S2_tscr.begin(), S2_tscr.end(), 0);
      tscr_count = 0;
    }

    float S1_sum, S2_sum;
    float * outdat = sums->get_dattfp();

    const unsigned int out_ndim = sums->get_ndim();
    assert (out_ndim == 2);

    switch (compute_input->get_order())
    {
      case dsp::TimeSeries::OrderTFP:
      {
        if (verbose) {
          cerr << "dsp::SpectralKurtosis::compute: OrderTFP" << endl;
        }
        const unsigned int chan_stride = nchan * npol * ndim;
        float * indat;

        for (unsigned ipart=0; ipart < npart; ipart++)
        {
          indat = (float *) compute_input->get_dattfp() + (overlap_offset * ipart * chan_stride);

          bool do_tscr = !detection_flags[1] && (ipart % noverlap == 0);
          if (do_tscr)
            tscr_count ++;

          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            for (unsigned ipol=0; ipol < npol; ipol++)
            {
              S1_sum = 0;
              S2_sum = 0;

              // Square Law Detect for S1 + S2
              for (unsigned i=0; i<M; i++)
              {
                float re = indat[chan_stride*i];
                float im = indat[chan_stride*i+1];
                float sqld = (re * re) + (im * im);
                S1_sum += sqld;
                S2_sum += (sqld * sqld);
              }

              unsigned out_index = ichan*npol + ipol;

              if (do_tscr)
              {
                // add the sums to the M timeseries
                S1_tscr [out_index] += S1_sum;
                S2_tscr [out_index] += S2_sum;
              }

              // store the S1 and S2 sums for later SK calculation
              if (S1_sum == 0)
                outdat[out_index*2] = outdat[out_index*2+1] = 0;
              else
              {
                outdat[out_index*2] = S1_sum;
                outdat[out_index*2+1] = S2_sum;
              }

              indat += ndim;
            }
          }

          outdat += nchan * npol * out_ndim;
        }
        break;
      }

      case dsp::TimeSeries::OrderFPT:
      {
        if (verbose) {
          cerr << "dsp::SpectralKurtosis::compute: OrderFPT" << endl;
        }
        const unsigned int nfloat = M * ndim;
        const unsigned int offset_nfloat = overlap_offset * ndim;

        // foreach computation
        for (unsigned ipart=0; ipart < npart; ipart++)
        {
          bool do_tscr = !detection_flags[1] && (ipart % noverlap == 0);
          if (do_tscr)
            tscr_count ++;

          for (unsigned ichan=0; ichan<nchan; ichan++)
          {
            for (unsigned ipol=0; ipol < npol; ipol++)
            {
              // input pointer for channel pol
              const float* indat = compute_input->get_datptr (ichan, ipol) + ipart * offset_nfloat;
              // cerr << "  ichan=" << ichan << ", ipol=" << ipol << ", ipart=" << ipart << endl;

              S1_sum = 0;
              S2_sum = 0;

              // WvS: the following loop assumes ndim == 2 (complex-valued data)

              // Square Law Detect for S1 + S2
              for (unsigned i=0; i<nfloat; i+=2)
              {
                // cerr << "  ichan=" << ichan << ", ipol=" << ipol << ", ipart=" << ipart;
                // cerr << " " << indat[i] << ", " << indat[i+1] << endl;
                float sqld = (indat[i] * indat[i]) + (indat[i+1] * indat[i+1]);
                S1_sum += sqld;
                S2_sum += (sqld * sqld);
              }

              unsigned out_index = ichan*npol + ipol;
              if (do_tscr)
              {
                S1_tscr [out_index] += S1_sum;
                S2_tscr [out_index] += S2_sum;
              }

              // store the S1 and S2 sums for later SK calculation
              if (S1_sum == 0)
                outdat[out_index*2] = outdat[out_index*2+1] = 0;
              else
              {
                outdat[out_index*2] = S1_sum;
                outdat[out_index*2+1] = S2_sum;
              }

            }
          }

          outdat += nchan * npol * out_ndim;
        }
        break;
      }

      default:
      {
        throw Error (InvalidState, "dsp::SpectralKurtosis::compute",
                     "unsupported input order");
      }
    }

    // calculate the SK Estimator for the whole block of data
    if (!detection_flags[1])
    {
      float M_t = M * tscr_count;
      float M_fac = (M_t+1) / (M_t-1);
      float * outdat = estimates_tscr->get_dattfp();
      if (verbose || debugd < 1)
        cerr << "dsp::SpectralKurtosis::compute tscr M=" << M_t <<" M_fac=" << M_fac << endl;
      for (unsigned ichan=0; ichan<nchan; ichan++)
      {
        for (unsigned ipol=0; ipol<npol; ipol++)
        {
          S1_sum = S1_tscr[ichan*npol + ipol];
          S2_sum = S2_tscr[ichan*npol + ipol];
          if (S1_sum == 0)
            outdat[ichan*npol + ipol] = 0;
          else
            outdat[ichan*npol + ipol] = M_fac * (M_t * (S2_sum / (S1_sum * S1_sum)) - 1);
        }
      }
    }
  }


  if (verbose || debugd < 1)
    cerr << "dsp::SpectralKurtosis::compute done" << endl;
  if (debugd < 1)
    debugd++;
}

void dsp::SpectralKurtosis::set_thresholds (float _std_devs)
{
  for (unsigned ires=0; ires < resolution.size(); ires++)
    resolution[ires].set_std_devs (_std_devs);
}

void dsp::SpectralKurtosis::Resolution::set_thresholds (float _std_devs,
                                                        bool verbose) const
{
  if (verbose)
    std::cerr << "dsp::SpectralKurtosis::Resolution::set_thresholds "
            "SKlimits(" << M << ", " << std_devs << ")" << endl;
  dsp::SKLimits limits(M, std_devs);
  limits.calc_limits();

  thresholds.resize(2);
  thresholds[0] = (float) limits.get_lower_threshold();
  thresholds[1] = (float) limits.get_upper_threshold();

  if (verbose)
    std::cerr << "dsp::SpectralKurtosis::Resolution::set_thresholds "
         "M=" << M << " std_devs=" << std_devs  << 
         " [" << thresholds[0] << " - " << thresholds[1] << "]" << endl;
}

void dsp::SpectralKurtosis::set_channel_range (unsigned start, unsigned end)
{
  resolution[0].add_include( pair<unsigned,unsigned> (start, end-1) );
}

void dsp::SpectralKurtosis::set_options (bool _disable_fscr,
    bool _disable_tscr, bool _disable_ft)
{
  detection_flags[0] = _disable_fscr;
  detection_flags[1] = _disable_tscr;
  detection_flags[2] = _disable_ft;
}

void dsp::SpectralKurtosis::detect ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect" << endl;

  if (verbose || debugd < 1)
  {
    cerr << "dsp::SpectralKurtosis::detect INPUT"
         << " nchan=" << nchan << " nbit=" << input->get_nbit()
         << " npol=" << npol << " ndim=" << ndim << endl;

    cerr << "dsp::SpectralKurtosis::detect OUTPUT"
         << " ndat=" << zapmask->get_ndat() << " nchan=" << zapmask->get_nchan()
         << " nbit=" << zapmask->get_nbit() << " npol=" << zapmask->get_npol()
         << " ndim=" << zapmask->get_ndim() << endl;
  }

  npart_total += (resolution.front().npart * nchan);

  // reset the mask to all 0 (no zapping)
  reset_mask();

  // apply the tscrunches SKFB estimates to the mask
  if (!detection_flags[1])
    detect_tscr ();
  if (report) {
    char_reporter.emit(
      "zapmask_tscr",
      zapmask->get_datptr(),
      zapmask->get_nchan(),
      zapmask->get_npol(),
      zapmask->get_ndat(),
      zapmask->get_ndim());
  }

  for (unsigned ires=0; ires < resolution.size(); ires++)
  {
    if (ires > 0)
      tscrunch_sums (resolution[ires-1], resolution[ires]);

    // apply the SKFB estimates to the mask
    if (!detection_flags[2])
      detect_skfb (ires);

    if (report )
    {
      char_reporter.emit(
      "zapmask_skfb",
      zapmask->get_datptr(),
      zapmask->get_nchan(),
      zapmask->get_npol(),
      zapmask->get_ndat(),
      zapmask->get_ndim());
    }

    if (!detection_flags[0])
      detect_fscr (ires);

    if (report)
    {
      char_reporter.emit(
      "zapmask_fscr",
      zapmask->get_datptr(),
      zapmask->get_nchan(),
      zapmask->get_npol(),
      zapmask->get_ndat(),
      zapmask->get_ndim());
    }

    if (ires == 0)
      count_zapped ();
  }

  if (debugd < 1)
    debugd++;
}

void dsp::SpectralKurtosis::tscrunch_sums (Resolution& from, Resolution& to)
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::tscrunch_sums from "
      "M=" << from.get_M() << " noverlap=" << from.noverlap << " to "
      "M=" << to.get_M() << " noverlap=" << to.noverlap << endl;

  Resolution convert;
  convert.set_M( to.get_M() / from.get_M() );
  convert.noverlap = to.noverlap;
  convert.prepare (sums->get_ndat());

  const unsigned nsum = convert.get_M();
  const uint64_t npart = convert.npart;
  const unsigned offset = convert.overlap_offset;

  if (verbose)
    cerr << "dsp::SpectralKurtosis::tscrunch_sums convert "
      "nsum=" << convert.get_M() << " offset=" << offset << " "
      "npart=" << npart << endl;

  unsigned sum_ndim = sums->get_ndim();
  assert (sum_ndim == 2);

  // in place tscrunch
  float* outdat = sums->get_dattfp();
  float* indat = outdat;

  const uint64_t fpd_blocksize = nchan * npol * sum_ndim;
  const uint64_t input_stride = fpd_blocksize * from.noverlap;
  const uint64_t input_offset = fpd_blocksize * offset;

  // compare SK estimator for each pol to expected values
  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    // for each channel and pol in the SKFB
    for (unsigned idat=0; idat < fpd_blocksize; idat++)
      outdat[idat] = indat[idat];

    for (unsigned isum=1; isum < nsum; isum++)
      for (unsigned idat=0; idat < fpd_blocksize; idat++)
        outdat[idat] += indat[isum*input_stride+idat];

    outdat += fpd_blocksize;
    indat += input_offset;
  }
}


/*
 * Use the tscrunched SK statistic from the SKFB to detect RFI on each channel
 */
void dsp::SpectralKurtosis::detect_tscr ()
{
  unsigned M = resolution.front().get_M();
  unsigned npart = resolution.front().npart;

  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_tscr(" << npart << ")" << endl;

  const float* indat    = estimates_tscr->get_dattfp();
  unsigned char* outdat = 0;
  unsigned zap_chan;
  float V;

  uint64_t m = M * tscr_count;
  float lower = 0;
  float upper = 0;

  M_tscr = float(m);

  if (npart)
  {
    bool must_compute = true;
    for (unsigned i=0; i<thresholds_tscr_m.size(); i++)
    {
      if (thresholds_tscr_m[i] == m)
      {
        must_compute = false;
        lower = thresholds_tscr_lower[i];
        upper = thresholds_tscr_upper[i];
        M_tscr = float(m);
      }
    }

    float std_devs = resolution.front().get_std_devs();

    if (must_compute)
    {
      if (verbose)
        cerr << "dsp::SpectralKurtosis::detect_tscr SKlimits(" << M_tscr << ", " << std_devs << ")" << endl;

      dsp::SKLimits limits(M_tscr, std_devs);
      limits.calc_limits();
      lower = float(limits.get_lower_threshold());
      upper = float(limits.get_upper_threshold());

      thresholds_tscr_m.push_back(m);
      thresholds_tscr_lower.push_back(lower);
      thresholds_tscr_upper.push_back(upper);
    }

    if (verbose)
      cerr << "dsp::SpectralKurtosis::detect_tscr M=" << M_tscr << " std_devs="
           << std_devs  << " [" << lower << " - " << upper << "]" << endl;
  }

  if (engine)
  {
    engine->detect_tscr (sums, estimates_tscr, zapmask, upper, lower);
    return;
  }

  const vector<bool>& channels = resolution.front().get_channels(nchan);
  
  for (uint64_t ichan=0; ichan < nchan; ichan++)
  {
    if (!channels[ichan])
      continue;
    
    zap_chan = 0;
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      V = indat[ichan*npol + ipol];
      if (V > upper || V < lower)
        zap_chan = 1;
    }

    if (zap_chan)
    {
      // if (verbose)
      //   cerr << "dsp::SpectralKurtosis::detect_tscr zap V=" << V << ", "
      //        << "ichan=" << ichan << endl;
      outdat = zapmask->get_datptr();
      for (unsigned ipart=0; ipart < npart; ipart++)
      {
        outdat[ichan] = 1;
        zap_counts[ZAP_TSCR]++;
        outdat += nchan;
      }
    }
  }
}

void dsp::SpectralKurtosis::detect_skfb (unsigned ires)
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_skfb(" << ires << ")" << endl;

  unsigned M = resolution[ires].get_M ();
  unsigned npart = resolution[ires].npart;

  const vector<float>& thresholds = resolution[ires].get_thresholds ();
  assert (thresholds.size() == 2);

  unsigned nflag = 1;
  unsigned flag_step = 1;
  unsigned flag_offset = 1;

  if (ires > 0)
  {
    nflag = M / resolution[0].get_M();
    flag_step = resolution[0].noverlap;
    flag_offset = nflag * flag_step / resolution[ires].noverlap;
  }

  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_skfb "
            "nflag=" << nflag << " step=" << flag_step << endl;

  if (engine)
  {
    engine->detect_ft (sums, zapmask, thresholds[1], thresholds[0]);
    return;
  }

  const float* indat    = sums->get_dattfp();
  unsigned char* outdat = zapmask->get_datptr();
  const unsigned sum_ndim = sums->get_ndim();
  assert (sum_ndim == 2);

  char zap;

  const float M_fac = (float)(M+1) / (M-1);

  const vector<bool>& channels = resolution[ires].get_channels(nchan);

  // compare SK estimator for each pol to expected values
  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    // for each channel and pol in the SKFB
    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      zap = 0;

      // only count skfb zapped channels in the in-band region
      bool count_zapped = (ires == 0) && channels[ichan];
 
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        unsigned index = (npol*ichan + ipol) * sum_ndim;
        float S1_sum = indat[index];
        float S2_sum = indat[index+1];

        float V = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);

        if (V > thresholds[1] || V < thresholds[0])
          zap = 1;
      }

      if (zap)
      {
        for (unsigned iflag=0; iflag < nflag; iflag++)
        {
          unsigned outdex = iflag*nchan*flag_step + ichan;
          outdat[outdex] = 1;

          if (count_zapped)
            zap_counts[ZAP_SKFB]++;
        }
      }
    }

    indat += nchan * npol * sum_ndim;
    outdat += nchan * flag_offset;
  }
}

void dsp::SpectralKurtosis::reset_mask ()
{
  if (engine)
  {
    engine->reset_mask (zapmask);
    return;
  }

  zapmask->zero();
}

void dsp::SpectralKurtosis::count_zapped ()
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::count_zapped hits=" 
         << unfiltered_hits << endl;

  const float * indat;
  unsigned char * outdat;

  if (engine)
  {
    int zapped = engine->count_mask (zapmask);
    indat = engine->get_estimates (sums);
    outdat = engine->get_zapmask(zapmask);
    zap_counts[ZAP_ALL] += zapped;
  }
  else
  {
    indat  = sums->get_dattfp();
    outdat = zapmask->get_datptr();
  }

  const unsigned sum_ndim = sums->get_ndim();
  assert (sum_ndim == 2);

  unsigned ires = 0;
  unsigned M = resolution[ires].get_M();
  unsigned npart = resolution[ires].npart;

  const float M_fac = (float)(M+1) / (M-1);

  assert (npart == sums->get_ndat());
  if (unfiltered_hits == 0)
  {
    filtered_sum.resize (npol * nchan);
    std::fill (filtered_sum.begin(), filtered_sum.end(), 0);

    filtered_hits.resize (nchan);
    std::fill (filtered_hits.begin(), filtered_hits.end(), 0);

    unfiltered_sum.resize (npol * nchan);
    std::fill (unfiltered_sum.begin(), unfiltered_sum.end(), 0);
  }

  const vector<bool>& channels = resolution[ires].get_channels(nchan);

  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    unfiltered_hits ++;

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      if (!channels[ichan])
	continue;
      
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        uint64_t index = ((ipart*nchan + ichan) * npol + ipol) * sum_ndim; 
        unsigned outdex = ichan * npol + ipol;

        float S1_sum = indat[index];
        float S2_sum = indat[index+1];

        float V = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);

        unfiltered_sum[outdex] += V;

        if (ipol == 0 && outdat[(ipart*nchan) + ichan] == 1)
        {
          zap_counts[ZAP_ALL] ++;
          continue;
        }

        filtered_sum[outdex] += V;
      }

      filtered_hits[ichan] ++;
    }
  }
}

void dsp::SpectralKurtosis::detect_fscr (unsigned ires)
{
  if (verbose)
    cerr << "dsp::SpectralKurtosis::detect_fscr(" << ires << ")" << endl;

  unsigned M = resolution[ires].get_M();
  unsigned npart = resolution[ires].npart;
  float std_devs = resolution[ires].get_std_devs();

  float _M = (float) M;
  float mu2 = (4 * _M * _M) / ((_M-1) * (_M + 2) * (_M + 3));

  if (engine)
  {
    // float one_sigma_idat   = sqrt(mu2 / (float) nchan);
    // const float upper = 1 + ((1+std_devs) * one_sigma_idat);
    // const float lower = 1 - ((1+std_devs) * one_sigma_idat);
    // cerr << "dsp::SpectralKurtosis::detect_fscr:" <<
    //   " upper=" << upper <<
    //   " lower=" << lower << endl;
    // engine->detect_fscr (sums, zapmask, lower, upper, channels[0], channels[1]);

    /*
      NZAPP-207: WvS 2020-05-06 GPU interface needs to be updated to receive
      a mask of bool for each channel, instead of a single start/end range
    */
    unsigned ichan_start = 0;
    unsigned ichan_end = nchan;
    
    engine->detect_fscr (sums, zapmask, mu2, std_devs, ichan_start, ichan_end);
    
    return;
  }

  const float * indat  = sums->get_dattfp();
  unsigned char * outdat = zapmask->get_datptr();

  const unsigned sum_ndim = sums->get_ndim();
  assert (sum_ndim == 2);

  const float M_fac = (float)(M+1) / (M-1);

  float sk_avg;
  unsigned sk_avg_cnt = 0;

  unsigned zap_ipart;
  uint64_t nzap = 0;

  unsigned nflag = 1;
  unsigned flag_step = 1;
  unsigned flag_offset = 1;

  if (ires > 0)
  {
    nflag = M / resolution[0].get_M();
    flag_step = resolution[0].noverlap;
    flag_offset = nflag * flag_step / resolution[ires].noverlap;
  }

  const vector<bool>& channels = resolution[ires].get_channels(nchan);
    
  // foreach SK integration
  for (uint64_t ipart=0; ipart < npart; ipart++)
  {
    zap_ipart = 0;
    for (unsigned ipol=0; ipol < npol; ipol++)
    {
      sk_avg = 0;
      sk_avg_cnt = 0;

      for (unsigned ichan=0; ichan < nchan; ichan++)
      {
        if (channels[ichan] && outdat[ichan] == 0)
        {
          unsigned index = (npol*ichan + ipol) * sum_ndim;
          float S1_sum = indat[index];
          float S2_sum = indat[index+1];

          float V = M_fac * (M * (S2_sum / (S1_sum * S1_sum)) - 1);

          sk_avg += V;
          sk_avg_cnt++;
        }
      }

      if (sk_avg_cnt > 0)
      {
        sk_avg /= (float) sk_avg_cnt;

        float one_sigma_idat = sqrt(mu2 / (float) sk_avg_cnt);
        float avg_upper_thresh = 1 + ((1+std_devs) * one_sigma_idat);
        float avg_lower_thresh = 1 - ((1+std_devs) * one_sigma_idat);
        if ((sk_avg > avg_upper_thresh) || (sk_avg < avg_lower_thresh))
        {
          if (verbose) {
            cerr << "Zapping ipart=" << ipart << " ipol=" << ipol << " sk_avg=" << sk_avg
                 << " [" << avg_lower_thresh << " - " << avg_upper_thresh
                 << "] cnt=" << sk_avg_cnt << endl;
          }
          zap_ipart = 1;
        }
      }
    }

    if (zap_ipart)
    {
      for (unsigned iflag=0; iflag < nflag; iflag++)
      {
        unsigned outdex = iflag*nchan*flag_step;
        for (unsigned ichan=0; ichan<nchan; ichan++)
          outdat[outdex + ichan] = 1;
      }
      if (ires > 0)
        zap_counts[ZAP_FSCR] += nchan;
      nzap += nchan;
    }

    indat += nchan * npol * sum_ndim;
    outdat += nchan * flag_offset;
  }
  //cerr << "dsp::SpectralKurtosis::detect_fscr ZAP=" << nzap << endl;
}


//! Perform the transformation on the input time series
void dsp::SpectralKurtosis::mask ()
{
  // indicate the output timeseries contains zeroed data
  output->set_zeroed_data (true);

  // resize the output to ensure the hits array is reallocated
  if (engine)
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::mask output->resize(" << output->get_ndat() << ")" << endl;
    output->resize (output->get_ndat());
  }

  // get base pointer to mask bitseries
  unsigned char * mask = zapmask->get_datptr ();

  unsigned M = resolution.front().get_M();
  // unsigned overlap = resolution.front().noverlap;
  unsigned overlap_offset = resolution.front().overlap_offset;

  // NZAPP-208 WvS: the most samples are lost at the most coarse resolution
  unsigned max_M = resolution.back().get_M();
  unsigned max_overlap_offset = resolution.back().overlap_offset;
  unsigned idat_start = max_M - max_overlap_offset;

#if _DEBUG
  cerr << "front.ndat=" << resolution.front().output_ndat << endl;
  cerr << "back.ndat=" << resolution.back().output_ndat << endl;
#endif

  // the number of fine blocks to skip
  unsigned nskip = idat_start / overlap_offset;
  // the number of fine blocks to process
  unsigned npart = (resolution.back().output_ndat - M) / overlap_offset + 1;

#if _DEBUG
  cerr << "front.npart=" << resolution.front().npart << endl;
  cerr << "back.npart=" << resolution.back().npart << endl;
  cerr << "npart=" << npart << " nskip=" << nskip << endl;
#endif

  assert ( idat_start % overlap_offset == 0 );
  assert ( (resolution.back().output_ndat - M) % overlap_offset == 0 );
  assert ( npart+nskip <= resolution.front().npart );

  assert ( idat_start + resolution.back().output_ndat < input->get_ndat() );
  assert ( resolution.back().output_ndat == output->get_ndat() );

  if (engine)
  {
    if (verbose)
      cerr << "dsp::SpectralKurtosis::transformation engine->setup(" << nchan << ")" << endl;
    engine->mask (zapmask, input, output, M);
  }
  else
  {
    // mask is a TFP ordered bit series, output is FPT order TimeSeries
    const unsigned nfloat = M * ndim;
    const unsigned int offset_nfloat = overlap_offset * ndim;

    for (unsigned ichan=0; ichan < nchan; ichan++)
    {
      for (unsigned ipol=0; ipol < npol; ipol++)
      {
        const float * indat  = input->get_datptr(ichan, ipol) + idat_start;
        float * outdat = output->get_datptr(ichan, ipol);
        for (uint64_t ipart=0; ipart < npart; ipart++)
        {
          if (mask[(ipart+nskip)*nchan+ichan])
          {
            for (unsigned j=0; j<nfloat; j++)
              outdat[j] = 0;
          }
          else
          {
            for (unsigned j=0; j<nfloat; j++)
              outdat[j] = indat[j];
          }

          indat += offset_nfloat;
          outdat += offset_nfloat;
        }
      }
    }
  }

  if (debugd < 1)
    debugd++;
}

//!
void dsp::SpectralKurtosis::insertsk ()
{
  if (engine)
    engine->insertsk (sums, output, resolution.front().get_M());
}

