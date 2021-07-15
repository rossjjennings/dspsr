/***************************************************************************
 *
 *   Copyright (C) 2015 by Matthew Kerr
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToFITS.h"

#include "dsp/IOManager.h"
#include "dsp/Unpacker.h"

#include "dsp/FITSFile.h"
#include "dsp/MultiFile.h"
#include "dsp/FITSUnpacker.h"

#include "dsp/TFPFilterbank.h"
#include "dsp/Filterbank.h"
#include "dsp/Detection.h"

#include "dsp/SampleDelay.h"
#include "dsp/DedispersionSampleDelay.h"
#include "dsp/DelayStartTime.h"

#include "dsp/FScrunch.h"
#include "dsp/TScrunch.h"
#include "dsp/PScrunch.h"
#include "dsp/PolnSelect.h"
#include "dsp/PolnReshape.h"

#include "dsp/FITSDigitizer.h"
#include "dsp/FITSOutputFile.h"

#if HAVE_CUDA
#include "dsp/ConvolutionCUDA.h"
#include "dsp/ConvolutionCUDASpectral.h"
#include "dsp/FilterbankEngineCUDA.h"
#include "dsp/OptimalFilterbank.h"
#include "dsp/TransferCUDA.h"
#include "dsp/DetectionCUDA.h"
#include "dsp/FScrunchCUDA.h"
#include "dsp/TScrunchCUDA.h"
#include "dsp/MemoryCUDA.h"
#include "dsp/TimeSeriesCUDA.h"
#include "dsp/SampleDelayCUDA.h"
#include "dsp/DelayStartTimeCUDA.h"
#include "dsp/FITSDigitizerCUDA.h"
#include "dsp/TransferBitSeriesCUDA.h"
#endif


using namespace std;

bool dsp::LoadToFITS::verbose = false;

static void* const undefined_stream = (void *) -1;

dsp::LoadToFITS::LoadToFITS (Config* configuration)
{
  kernel = NULL;
  set_configuration (configuration);
}

//! Run through the data
void dsp::LoadToFITS::set_configuration (Config* configuration)
{
  SingleThread::set_configuration (configuration);
  config = configuration;
}

dsp::LoadToFITS::Config::Config()
{
  can_cuda = true;
  can_thread = true;

  // block size in MB
  block_size = 2.0;

  order = dsp::TimeSeries::OrderTFP;

  // by default, do not denormalize using DAT_SCL and DAT_OFFS
  apply_FITS_scale_and_offset = false;

  filterbank.set_nchan(0);
  filterbank.set_freq_res(0);
  filterbank.set_convolve_when(Filterbank::Config::Never);

  maximum_RAM = 256 * 1024 * 1024;
  times_minimum_ndat = 1;

  dispersion_measure = 0;
  dedisperse = false;
  coherent_dedisp = false;
  start_time_delay = 0;

  rescale_seconds = -1;
  rescale_constant = false;

  integration_length = 0;

  nbits = 2;

  npol = 4;

  nsblk = 2048;
  tsamp = 64e-6;

  // by default, time series weights are not used
  weighted_time_series = false;

  // by default, lower sideband output
  upper_sideband_output = false;
}

// set block size to this factor times the minimum possible
void dsp::LoadToFITS::Config::set_times_minimum_ndat (unsigned ndat)
{
  times_minimum_ndat = ndat;
  maximum_RAM = 0;
}

// set block_size to result in approximately this much RAM usage
void dsp::LoadToFITS::Config::set_maximum_RAM (uint64_t ram)
{
  times_minimum_ndat = 1;
  maximum_RAM = ram;
}

void dsp::LoadToFITS::Config::set_quiet ()
{
  SingleThread::Config::set_quiet();
  LoadToFITS::verbose = false;
}

void dsp::LoadToFITS::Config::set_verbose ()
{
  SingleThread::Config::set_verbose();
  LoadToFITS::verbose = true;
}

void dsp::LoadToFITS::Config::set_very_verbose ()
{
  SingleThread::Config::set_very_verbose();
  LoadToFITS::verbose = 3;
  FITSOutputFile::verbose = 3;
}

void dsp::LoadToFITS::construct () try
{
  // sets operations to zero length then adds IOManger/unpack
  SingleThread::construct ();

  bool run_on_gpu = false;
#if HAVE_CUDA
  run_on_gpu = thread_id < config->get_cuda_ndevice();
  cudaStream_t stream = reinterpret_cast<cudaStream_t>( gpu_stream );
#endif

  /*
    The following lines "wire up" the signal path, using containers
    to communicate the data between operations.
  */

  // set up for optimal memory usage pattern

  Unpacker* unpacker = manager->get_unpacker();

#if HAVE_CFITSIO && HAVE_fits

  if (config->apply_FITS_scale_and_offset &&
      manager->get_info()->get_machine() == "FITS")
  {
    FITSUnpacker* fun = dynamic_cast<FITSUnpacker*> (manager->get_unpacker());
    fun->apply_scale_and_offset (true);
  }

#endif

  if (!config->dedisperse && unpacker->get_order_supported (config->order))
    unpacker->set_output_order (config->order);

  // get basic information about the observation

  Observation* obs = manager->get_info();
  const unsigned nchan = obs->get_nchan ();
  const unsigned npol = obs->get_npol ();
  const unsigned ndim = obs->get_ndim ();
  const double rate = obs->get_rate () ;

  if (verbose)
  {
    cerr << "Source = " << obs->get_source() << endl;
    cerr << "Frequency = " << obs->get_centre_frequency() << endl;
    cerr << "Bandwidth = " << obs->get_bandwidth() << endl;
    cerr << "Channels = " << nchan << endl;
    cerr << "Sampling rate = " << rate << endl;
    cerr << "State = " << tostring(obs->get_state()) <<endl;
  }

  obs->set_dispersion_measure( config->dispersion_measure );

  unsigned fb_nchan = config->filterbank.get_nchan();
  unsigned nsample;
  double tsamp, samp_per_fb;
  unsigned tres_factor;
  double factor = obs->get_state() == Signal::Nyquist? 0.5 : 1.0;

  if (fb_nchan > 0)
  {
    // Strategy will be to tscrunch from Nyquist resolution to desired reso.
    // voltage samples per filterbank sample
    samp_per_fb = config->tsamp * rate;
    if (verbose)
      cerr << "voltage samples per filterbank sample=" << samp_per_fb << endl;
    // correction for number of samples per filterbank channel
    unsigned effective_channelisation = fb_nchan / nchan;
    tres_factor = round(factor*samp_per_fb/effective_channelisation);
    tsamp = tres_factor/factor * (fb_nchan/nchan)/rate;

    // voltage samples per output block
    nsample = round(samp_per_fb * config->nsblk);
  }
  else
  {
    samp_per_fb = 1.0;
    tres_factor = round(rate * config->tsamp);
    tsamp = tres_factor/factor * 1/rate;
    nsample = config->nsblk * tres_factor;
  }

  cerr << "digifits: requested tsamp=" << config->tsamp << " rate=" << rate << endl
       << "             actual tsamp=" << tsamp << " (tscrunch=" << tres_factor << ")" << endl;
  if (verbose)
    cerr << "digifits: nsblk=" << config->nsblk << endl;

  // the unpacked input will occupy nbytes_per_sample
  double nbytes_per_sample = sizeof(float) * nchan * npol * ndim;
  double MB = 1024.0 * 1024.0;

  // ideally, block size would be a full output block, but this is too large
  // pick a nice fraction that will divide evently into maximum RAM
  // NB this doesn't account for copies (yet)

  if (verbose)
    cerr << "digifits: nsample * nbytes_per_sample=" << nsample * nbytes_per_sample
         << " config->maximum_RAM=" << config->maximum_RAM << endl;
  while (nsample * nbytes_per_sample > config->maximum_RAM) nsample /= 2;

  if (verbose)
    cerr << "digifits: block_size=" << (nbytes_per_sample*nsample)/MB
         << " MB " << "(" << nsample << " samp)" << endl;

  manager->set_block_size ( nsample );

  // if running on multiple GPUs, make nsblk such that no buffering is
  // required
  if ((run_on_gpu) and (config->get_total_nthread() > 1))
  {
    config->nsblk = nsample / samp_per_fb;
    if (verbose)
      cerr << "digifits: due to GPU multi-threading, setting nsblk="<<config->nsblk << endl;
  }

  TimeSeries* timeseries = unpacked;

#if HAVE_CUDA
  if (run_on_gpu)
  {
    timeseries->set_memory (device_memory);
    timeseries->set_engine (new CUDA::TimeSeriesEngine (device_memory));
  }
#endif

  if (!obs->get_detected())
  {
    cerr << "digifits: input data not detected" << endl;

    // if no filterbank specified
    if ((fb_nchan == 0) && (nchan == 1))
    {
      throw Error(InvalidParam,"dsp::LoadToFITS::construct",
          "must specify filterbank scheme if single channel data");
    }

    if ((config->coherent_dedisp) && (config->dispersion_measure != 0.0))
    {
      cerr << "digifits: performing coherent dedispersion at " << config->dispersion_measure << endl;
      kernel = new Dedispersion;
      kernel->set_dispersion_measure( config->dispersion_measure );

      unsigned frequency_resolution = config->filterbank.get_freq_res ();
      if (frequency_resolution)
      {
        cerr << "digifits: setting filter length to " << frequency_resolution << endl;
        kernel->set_frequency_resolution (frequency_resolution);
        //kernel -> set_times_minimum_nfft (frequency_resolution);
      }
    }
    else
    {
      if (config->dispersion_measure != 0.0)
        cerr << "digifits: performing incoherent dedispersion" << endl;
      config->coherent_dedisp = false;
    }

    // filterbank is performing channelisation
    if (config->filterbank.get_nchan() > 1)
    {
      // If user specifies -FN:D, enable coherent dedispersion
      if (config->filterbank.get_convolve_when() == Filterbank::Config::During)
      {
        // during is the only option for filterbank
        config->filterbank.set_convolve_when( Filterbank::Config::During );
      }
      else
      {
        config->coherent_dedisp = false;
      }

#if HAVE_CUDA
      if (run_on_gpu)
      {
        config->filterbank.set_device ( device_memory.ptr() );
        config->filterbank.set_stream ( gpu_stream );
      }
#endif

      filterbank = config->filterbank.create ();

      filterbank->set_nchan( config->filterbank.get_nchan() );
      filterbank->set_input( timeseries );
      filterbank->set_output( timeseries = new_TimeSeries() );

#if HAVE_CUDA
      if (run_on_gpu)
        timeseries->set_memory (device_memory);
#endif

      if (config->coherent_dedisp && kernel)
        filterbank->set_response( kernel );

      if ( !config->coherent_dedisp )
      {
        unsigned freq_res = config->filterbank.get_freq_res();
        if (freq_res > 1)
          filterbank->set_frequency_resolution ( freq_res );
      }

      operations.push_back( filterbank.get() );
    }

    // if convolution does not happen during filterbanking
    if (config->coherent_dedisp && config->filterbank.get_convolve_when() != Filterbank::Config::During)
    {
      cerr << "digifits: creating convolution operation" << endl;

      if (!convolution)
        convolution = new Convolution;

      if (!config->input_buffering)
        convolution->set_buffering_policy (NULL);

      convolution->set_response (kernel);
      //if (!config->integration_turns)
      //  convolution->set_passband (passband);

      convolution->set_input  (timeseries);
      convolution->set_output (timeseries = new_TimeSeries() );    // out of place

#if HAVE_CUDA
      if (run_on_gpu)
      {
        timeseries->set_memory (device_memory);
        convolution->set_device (device_memory.ptr());
        unsigned nchan = manager->get_info()->get_nchan();
        if (fb_nchan)
          nchan *= fb_nchan;
        if (nchan >= 16)
          convolution->set_engine (new CUDA::ConvolutionEngineSpectral (stream));
        else
          convolution->set_engine (new CUDA::ConvolutionEngine (stream));
      }
#endif

      operations.push_back (convolution.get());
    }

    if (verbose)
	    cerr << "digifits: creating detection operation" << endl;
    Detection* detection = new Detection;
    detection->set_input ( timeseries );
    detection->set_output ( timeseries = new_TimeSeries() );

    // force out of place detection operation
    if (config->npol == 1)
      detection->set_output_state (Signal::Intensity);
    else if (config->npol == 2)
      detection->set_output_state (Signal::PPQQ);
    else if (config->npol == 4)
      detection->set_output_state (Signal::Coherence);
    else
      throw Error (InvalidState, "dsp::LoadToFITS::construct",
                   "output polarisation invalid %d", config->npol);
#ifdef HAVE_CUDA
    if (run_on_gpu)
    {
      detection->set_engine (new CUDA::DetectionEngine(stream) );
      timeseries->set_memory (device_memory);
    }
#endif

    operations.push_back( detection );
  }

#if HAVE_CUDA
  if (run_on_gpu)
  {
    // to support faster input buffering
    timeseries->set_engine (new CUDA::TimeSeriesEngine (device_memory));
  }
#endif

  if ( config->dedisperse  || (config->coherent_dedisp && config->fscrunch_factor) )
  {
    if (verbose)
      cerr << "digifits: removing dispersion delays" << endl;

    SampleDelay* delay = new SampleDelay;

    delay->set_input (timeseries);
    delay->set_output (timeseries = new_TimeSeries());
    delay->set_function (new Dedispersion::SampleDelay);

    // coherent dedispersion + fscrunching requires additional sample delays
    if (config->coherent_dedisp && config->fscrunch_factor)
      delay->set_delay_span(config->fscrunch_factor);

#if HAVE_CUDA
    if (run_on_gpu)
    {
      delay->set_engine (new CUDA::SampleDelayEngine (stream));
      timeseries->set_memory (device_memory);
      timeseries->set_engine (new CUDA::TimeSeriesEngine (device_memory));
    }
#endif

    operations.push_back( delay );
  }

  if (config->start_time_delay > 0)
  {
    MJD start = obs->get_start_time();
    MJD new_start = start + config->start_time_delay;
    MJD delay = new_start - start;
    cerr << "obs start time=" << start.in_seconds() << " seconds" << endl;
    if (verbose)
      cerr << "digifits: delaying start time by " << delay.in_seconds() << endl;

    DelayStartTime * start_delay = new DelayStartTime;
    start_delay->set_input (timeseries);
    start_delay->set_output (timeseries);
    start_delay->set_start_time (new_start);
#if HAVE_CUDA
    if (run_on_gpu)
    {
      start_delay->set_engine (new CUDA::DelayStartTimeEngine (stream));
      timeseries->set_memory (device_memory);
      timeseries->set_engine (new CUDA::TimeSeriesEngine (device_memory));
    }
#endif
    operations.push_back( start_delay );
  }

  if (tres_factor > 1)
  {
    TScrunch* tscrunch = new TScrunch;
    tscrunch->set_factor ( tres_factor );
    tscrunch->set_input ( timeseries );
    tscrunch->set_output ( timeseries = new_TimeSeries() );

#if HAVE_CUDA
    if ( run_on_gpu )
    {
      tscrunch->set_engine ( new CUDA::TScrunchEngine(stream) );
      timeseries->set_memory (device_memory);
      timeseries->set_engine (new CUDA::TimeSeriesEngine (device_memory));
    }
#endif
    operations.push_back( tscrunch );
  }

  if ( config->fscrunch_factor )
  {
    FScrunch* fscrunch = new FScrunch;

    fscrunch->set_factor( config->fscrunch_factor );
    fscrunch->set_input( timeseries );
    fscrunch->set_output( timeseries = new_TimeSeries() );

#if HAVE_CUDA
    if (run_on_gpu)
    {
      fscrunch->set_engine (new CUDA::FScrunchEngine (stream));
      timeseries->set_memory (device_memory);
      timeseries->set_engine (new CUDA::TimeSeriesEngine (device_memory));
    }
#endif

    operations.push_back( fscrunch );
  }

  if (verbose)
    cerr << "digifits: creating output bitseries container" << endl;

  BitSeries* bitseries = new BitSeries;

  if (verbose)
    cerr << "digifits: creating PSRFITS digitizer with nbit="
         << config->nbits << endl;

  FITSDigitizer* digitizer = new FITSDigitizer (config->nbits);
  digitizer->set_input (timeseries);
  digitizer->set_output (bitseries);
  digitizer->set_upper_sideband_output (config->upper_sideband_output);

#if HAVE_CUDA
  if (run_on_gpu)
  {
    Scratch* gpu_scratch = new Scratch;
    gpu_scratch->set_memory (device_memory);
    CUDA::FITSDigitizerEngine * fits_digitizer_engine = new CUDA::FITSDigitizerEngine(stream);
    fits_digitizer_engine->set_scratch (gpu_scratch);
    digitizer->set_engine ( fits_digitizer_engine );
    bitseries->set_memory (device_memory);
  }
#endif

  // PSRFITS allows us to save the reference spectrum in each output block
  // "subint", so we can take advantage of this to store the exect
  // reference spectrum for later use.  By default, we will rescale the
  // spectrum using values for exactly one block (nsblk samples).  This
  // potentially improves the dynamic range, but makes the observaiton more
  // subject to transiennts.  By calling set_rescale_nblock(N), the path
  // will keep a running mean/scale for N sample blocks.  This is presented
  // to the user through rescale_seconds, which will choose the appropriate
  // block length to approximate the requested time interval.
  digitizer->set_rescale_samples (config->nsblk);
  if (config->rescale_constant)
  {
    cerr << "digifits: holding scales and offsets constant" << endl;
    digitizer->set_rescale_constant (true);
  }
  else if (config->rescale_seconds > 0)
  {
    double tblock = config->tsamp * config->nsblk;
    unsigned nblock = unsigned ( config->rescale_seconds/tblock + 0.5 );
    if (nblock < 1) nblock = 1;
    digitizer->set_rescale_nblock (nblock);
    cerr << "digifits: using "<<nblock<<" blocks running mean for scales and constant ("<<tblock*nblock<<") seconds"<<endl;
#if HAVE_CUDA
    if (run_on_gpu)
      Error (InvalidParam, "dsp::LoadToFITS::construct",
             "rescale seconds not supported in CUDA engine");
#endif
  }

  operations.push_back( digitizer );

#if HAVE_CUDA
  if (run_on_gpu)
  {
    TransferBitSeriesCUDA* transfer = new TransferBitSeriesCUDA (stream);
    transfer->set_kind (cudaMemcpyDeviceToHost);
    transfer->set_input( bitseries );
    transfer->set_output( bitseries = new BitSeries() );
    bitseries->set_memory( new  CUDA::PinnedMemory() );
    operations.push_back (transfer);
  }
#endif

  if (verbose)
    cerr << "digifits: creating PSRFITS output file" << endl;

  const char* output_filename = 0;
  if (!config->output_filename.empty())
    output_filename = config->output_filename.c_str();

  FITSOutputFile* outputfile = new FITSOutputFile (output_filename);
  outputfile->set_nsblk (config->nsblk);
  outputfile->set_nbit (config->nbits);
  outputfile->set_max_length (config->integration_length);
  outputfile->set_upper_sideband_output (config->upper_sideband_output);
  outputFile = outputfile;
  outputFile->set_input (bitseries);

  operations.push_back( outputFile.get() );

  // add a callback for the PSRFITS reference spectrum
  digitizer->update.connect (
      dynamic_cast<FITSOutputFile*> (outputFile.get()),
      &FITSOutputFile::set_reference_spectrum);
}
catch (Error& error)
{
  throw error += "dsp::LoadToFITS::construct";
}

void dsp::LoadToFITS::prepare () try
{
  SingleThread::prepare();

  unsigned freq_res = config->coherent_dedisp? kernel->get_frequency_resolution() : config->filterbank.get_freq_res();
  if (freq_res == 0) freq_res = 1;
  if (config->filterbank.get_nchan())
    cerr << "digifits: creating " << config->filterbank.get_nchan()
         << " by " << freq_res << " back channel filterbank" << endl;
  else
    cerr << "digifits: processing " << manager->get_info()->get_nchan() << " channels" << endl;

  // TODO -- set an optimal block size for search mode
  minimum_samples = 0;
  unsigned block_overlap = 0;
  bool report_vitals = thread_id==0 && config->report_vitals;

  if (config->coherent_dedisp && kernel && report_vitals)
    cerr << "digifits: dedispersion filter length=" << kernel->get_ndat ()
         << " (minimum=" << kernel->get_minimum_ndat () << ")"
         << " complex samples" << endl;

  // Check that block size is sufficient for the filterbanks,
  // increase it if not.
  if (filterbank)
  {
    minimum_samples = filterbank->get_minimum_samples();
    if (report_vitals)
    {
      cerr << "digifits: " << config->filterbank.get_nchan() << " channel ";
      if (config->coherent_dedisp &&
          config->filterbank.get_convolve_when() == Filterbank::Config::During)
        cerr << "dedispersing ";
      else if (filterbank->get_freq_res() > 1)
        cerr << "by " << filterbank->get_freq_res() << " back ";

      cerr << "filterbank requires " << minimum_samples << " samples" << endl;

      if (!config->input_buffering)
      {
        block_overlap = filterbank->get_minimum_samples_lost ();
        if (Operation::verbose)
          cerr << "filterbank loses " << block_overlap << " samples" << endl;
      }
    }

    // need filterbank->get_minimum_samples samples
    manager->set_block_size (minimum_samples);

#if 0
    if (verbose)
      cerr << "digifits: filterbank minimum samples = "
        << filterbank->get_minimum_samples()
        << endl;

    if (filterbank->get_minimum_samples() >
        manager->get_input()->get_block_size())
    {
      cerr << "digifits: increasing data block size from "
        << manager->get_input()->get_block_size()
        << " to " << filterbank->get_minimum_samples()
        << " samples" << endl;
      manager->set_block_size( filterbank->get_minimum_samples() );
    }
#endif
  }

  unsigned filterbank_resolution = minimum_samples - block_overlap;

  if (convolution)
  {
    const Observation* info = manager->get_info();
    unsigned fb_factor = convolution->get_input()->get_nchan() * 2;
    fb_factor /= info->get_nchan() * info->get_ndim();

    minimum_samples = convolution->get_minimum_samples () * fb_factor;
    cerr << "digifits: convolution requires at least "
         << minimum_samples << " samples" << endl;

    if (!config->input_buffering)
    {
      block_overlap = convolution->get_minimum_samples_lost () * fb_factor;
      if (Operation::verbose)
        cerr << "convolution loses " << block_overlap << " samples" << endl;

      manager->set_filterbank_resolution (filterbank_resolution);
    }
  }

  uint64_t block_size = ( minimum_samples - block_overlap )
    * config->get_times_minimum_ndat() + block_overlap;

  // set the block size to at least minimum_samples
  if (block_size == 0)
    manager->set_maximum_RAM( config->get_maximum_RAM() );
  manager->set_copies( config->get_nbuffers() );

  manager->set_overlap( block_overlap );

  uint64_t ram = manager->set_block_size( block_size );

  if (report_vitals)
  {
    double megabyte = 1024*1024;
    cerr << "digifits: blocksize=" << manager->get_input()->get_block_size()
         << " samples or " << double(ram)/megabyte << " MB" << endl;
  }

}
catch (Error& error)
{
  throw error += "dsp::LoadToFITS::prepare";
}
