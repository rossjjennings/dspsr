//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2007 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

// dspsr/Signal/Pulsar/dsp/LoadToFoldConfig.h

#ifndef __baseband_dsp_LoadToFoldConfig_h
#define __baseband_dsp_LoadToFoldConfig_h

#include <string>

#include "dsp/LoadToFold1.h"
#include "dsp/FilterbankConfig.h"
#include "dsp/InverseFilterbankConfig.h"

namespace Pulsar
{
  class Parameters;
  class Predictor;
}

namespace dsp {

  //! Load, unpack, process and fold data into phase-averaged profile(s)
  class LoadToFold::Config : public SingleThread::Config
  {

    // set block size to this factor times the minimum possible
    unsigned times_minimum_ndat;

    // set block size to result in approximately this much RAM usage
    uint64_t maximum_RAM;

    // set block size to result in at least this much RAM usage
    uint64_t minimum_RAM;

  public:

    //! Default constructor
    Config ();

    // set block size to this factor times the minimum possible
    void set_times_minimum_ndat (unsigned);
    unsigned get_times_minimum_ndat () const { return times_minimum_ndat; }

    // set block_size to result in approximately this much RAM usage
    void set_maximum_RAM (uint64_t);
    uint64_t get_maximum_RAM () const { return maximum_RAM; }

    // set block_size to result in at least this much RAM usage
    void set_minimum_RAM (uint64_t);
    uint64_t get_minimum_RAM () const { return minimum_RAM; }

    // set the name of the archive class to be used for output
    void set_archive_class (const std::string&);

    // number of time samples used to estimate undigitized power
    unsigned excision_nsample;
    // cutoff power used for impulsive interference rejection
    float excision_cutoff;
    // sampling threshold
    float excision_threshold;

    // when unpacking FITS data, denormalize using DAT_SCL and DAT_OFFS
    bool apply_FITS_scale_and_offset;

    // perform coherent dedispersion
    bool coherent_dedispersion;

    // remove inter-channel dispersion delays
    bool interchan_dedispersion;

    // dispersion measure used in coherent dedispersion
    double dispersion_measure;

    // zap RFI during convolution
    bool zap_rfi;

    // use FFT benchmarks to choose an optimal FFT length
    bool use_fft_bench;

    // optimize the order in which data are stored (e.g. FPT vs TFP)
    bool optimal_order;

    // perform phase-coherent matrix convolution (calibration)
    std::string calibrator_database_filename;

    // set fft lengths and convolution edge effect lengths
    unsigned nsmear;
    unsigned times_minimum_nfft;

    // phase-locked filterbank phase bins
    unsigned plfb_nbin;
    // phase-locked filterbank channels
    unsigned plfb_nchan;

    // cyclic spectrum options
    unsigned cyclic_nchan;
    unsigned cyclic_mover;

    // compute and fold the fourth moments of the electric field
    bool fourth_moment;

    // compute and output mean and variance for pdmp
    bool pdmp_output;

    // apply spectral kurtosis filterbank
    bool sk_zap;

    // load spectral kurtosis configuration from YAML file
    std::string sk_config;

    // also produce the non-zapped version of the output
    bool nosk_too;

    // spectral kurtosis integration factor
    std::vector<unsigned> sk_m;
    void set_sk_m (std::string txt);

    // spectral kurtosis overlap factor
    std::vector<unsigned> sk_noverlap;
    void set_sk_noverlap (std::string txt);

    // number of stddevs to use for spectral kurtosis excision
    std::vector<float> sk_std_devs;
    void set_sk_std_devs (std::string txt);

    // first channel to begin SK Detection
    unsigned sk_chan_start;

    // last channel to conduct SK Detection
    unsigned sk_chan_end;

    // to disable SKDetector Fscrunch feature
    bool sk_no_fscr;

    // to disable SKDetector Tscrunch feature
    bool sk_no_tscr;

    // to disable SKDetector FT feature
    bool sk_no_ft;

    // number of CPU threads for spectral kurtosis filterbank
    unsigned sk_nthreads;

    // fold the spectral kurtosis filterbank output
    bool sk_fold;

    unsigned npol;
    unsigned nbin;
    unsigned ndim;

    // Filterbank configuration options
    Filterbank::Config filterbank;

    // Inverse Filterbank configuration options
    InverseFilterbank::Config inverse_filterbank;

    bool is_inverse_filterbank;

    // whether or not to apply deripple correction
    bool do_deripple;

    std::string inverse_filterbank_fft_window;

    bool force_sensible_nbin;

    // length of sub-integrations in pulse periods
    unsigned integration_turns;

    // length of sub-integrations in seconds
    double integration_length;

    // reference epoch = start of first sub-integration
    std::string integration_reference_epoch;

    // minimum sub-integration length written to disk
    double minimum_integration_length;

    // all sub-integrations written to a single file
    bool single_archive;

    // number of sub-integrations written to a single file
    unsigned subints_per_archive;

    // file naming convention
    std::string filename_convention;

    void single_pulse()
    {
      integration_turns = 1;
      integration_length = 0;
    }

    /*
      If multiple sub-integrations will be combined in a single archive,
      then a single archiver will be required to manage the integration
    */
    bool single_archiver_required ()
    {
      return single_archive || subints_per_archive > 0;
    }

    // multiple threads can (and should) write to disk at once
    bool concurrent_archives ()
    {
      return integration_turns && !single_archiver_required();
    }

    // there may be more than one file per UTC second
    bool may_be_more_than_one_archive_per_second ()
    {
      return integration_turns;
    }

    std::string reference_epoch;
    double reference_phase;
    double folding_period;
    bool   fractional_pulses;

    bool asynchronous_fold;

    /* There are three ways to fold multiple pulsars:

    1) give names: Fold will generate ephemeris and predictor
    2) give ephemerides: Fold will generate predictors
    3) give predictors: Fold will use them

    You may specify any combination of the above, but the highest numbered
    information will always be used.

    */

    // additional pulsar names to be folded
    std::vector< std::string > additional_pulsars;

    // the parameters of multiple pulsars to be folded
    std::vector< Reference::To<Pulsar::Parameters> > ephemerides;

    // the predictors of multiple pulsars to be folded
    std::vector< Reference::To<Pulsar::Predictor> > predictors;

    //! Return the number of folds to perform
    unsigned get_nfold () const;

    // don't output dynamic extensions in the file
    bool no_dynamic_extensions;

    // name of the output archive class
    std::string archive_class;
    bool archive_class_specified_by_user;

    // name of the output archive file
    std::string archive_filename;

    // extension appended to the output archive filename
    std::string archive_extension;

    // output archive post-processing jobs
    std::vector<std::string> jobs;

    //! Operate in quiet mode
    virtual void set_quiet ();

    //! Operate in verbose mode
    virtual void set_verbose ();

    //! Operate in very verbose mode
    virtual void set_very_verbose ();
  };

}

#endif
