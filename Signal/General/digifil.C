/***************************************************************************
 *
 *   Copyright (C) 2008 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/*
  digifil converts any file format recognized by dspsr into sigproc
  filterbank (.fil) format.
 */

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "dsp/LoadToFil.h"
#include "dsp/LoadToFilN.h"
#include "dsp/FilterbankConfig.h"

#include "CommandLine.h"
#include "FTransform.h"

#include <stdlib.h>

using namespace std;

// The LoadToFil configuration parameters
Reference::To<dsp::LoadToFil::Config> config;

// names of data files to be processed
vector<string> filenames;

void parse_options (int argc, char** argv);

int main (int argc, char** argv) try
{
  config = new dsp::LoadToFil::Config;

  parse_options (argc, argv);

  Reference::To<dsp::Pipeline> engine;
  if (config->get_total_nthread() > 1)
    engine = new dsp::LoadToFilN (config);
  else
    engine = new dsp::LoadToFil (config);

  engine->set_input( config->open (argc, argv) );
  engine->construct ();   
  engine->prepare ();   
  engine->run();
  engine->finish();
}
catch (Error& error)
{
  cerr << error << endl;
  return -1;
}

void parse_options (int argc, char** argv) try
{
  CommandLine::Menu menu;
  CommandLine::Argument* arg;
  
  menu.set_help_header ("digifil - convert dspsr input to search mode output");
  menu.set_version ("digifil " + tostring(dsp::version) +
		    " <" + FTransform::get_library() + ">");

  config->add_options (menu);

  // Need to rename default threading option due to conflict
  // with original digifil -t (time average) setting.
  arg = menu.find("t");
  arg->set_short_name('\0'); 
  arg->set_long_name("threads");
  arg->set_type("nthread");

  arg = menu.add (config->nbits, 'b', "bits");
  arg->set_help ("number of bits per output sample (-32 -> float)");
  arg->set_long_help
    ("Use '-b -32' to output samples as 32-bit floating point values");

  arg = menu.add (config->block_size, 'B', "MB");
  arg->set_help ("block size in megabytes");

  arg = menu.add (config->rescale_constant, 'c');
  arg->set_help ("keep offset and scale constant");

  arg = menu.add (config->excision_enable, '2');
  arg->set_help ("disable 2-bit excision");

  arg = menu.add (config->filterbank, 'F', "nchan[:D]");
  arg->set_help ("create a filterbank (voltages only)");
  arg->set_long_help
    ("Specify number of filterbank channels; e.g. -F 256\n"
     "Select coherently dedispersing filterbank with -F 256:D\n"
     "Set leakage reduction factor with -F 256:<N>\n");

  arg = menu.add (&config->filterbank, 
      &dsp::Filterbank::Config::set_freq_res, 
      'x', "nfft");
  arg->set_help ("backward FFT length in voltage filterbank");

  arg = menu.add (config->dedisperse, 'K');
  arg->set_help ("remove inter-channel dispersion delays");

  arg = menu.add (config->dispersion_measure, 'D', "dm");
  arg->set_help ("set the dispersion measure");

  arg = menu.add (config->tscrunch_factor, 't', "nsamp");
  arg->set_help ("decimate in time");

  arg = menu.add (config->fscrunch_factor, 'f', "nchan");
  arg->set_help ("decimate in frequency");

  arg = menu.add (config->npol, 'd', "npol");
  arg->set_help ("1=PP+QQ, 2=PP,QQ, 3=(PP+QQ)^2 4=PP,QQ,PQ,QP");

  arg = menu.add (config->poln_select, 'P', "ipol");
  arg->set_help ("process only a single polarization of input");

  arg = menu.add (config->rescale_seconds, 'I', "secs");
  arg->set_help ("rescale interval in seconds (0 -> disable rescaling)");
  arg->set_long_help
    ("Use '-I 0' to disable rescaling (useful in combination with '-b -32')");

  arg = menu.add (config->scale_fac, 's', "fac");
  arg->set_help ("data scale factor to apply");

  arg = menu.add (config->apply_FITS_scale_and_offset, "scloffs");
  arg->set_help ("denormalize using DAT_SCL and DAT_OFFS [PSRFITS]");

  arg = menu.add (config->output_filename, 'o', "file");
  arg->set_help ("output filename");

  bool revert = false;
  arg = menu.add (revert, 'p');
  arg->set_help ("revert to FPT order");

  menu.parse (argc, argv);

  if (revert)
    config->order = dsp::TimeSeries::OrderFPT;
}
catch (Error& error)
{
  cerr << error << endl;
  exit (-1);
}
catch (std::exception& error)
{
  cerr << error.what() << endl;
  exit (-1);
}

