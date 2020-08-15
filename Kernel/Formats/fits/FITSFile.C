//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <fcntl.h>
#include <assert.h>

#include "Pulsar/Pulsar.h"
#include "Pulsar/Archive.h"
#include "Pulsar/Receiver.h"
#include "Pulsar/Backend.h"
#include "Pulsar/FITSSUBHdrExtension.h"

#include "psrfitsio.h"
#include "fits_params.h"

#include "dsp/FITSFile.h"
#include "dsp/FITSOutputFile.h"
#include "dsp/CloneArchive.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using Pulsar::warning;


dsp::FITSFile::FITSFile (const char* filename)
  : File("FITSFile")
{
  zero_off = 0.0;
  current_byte = 0;
  current_row = 0;
  fp = NULL;
}

bool dsp::FITSFile::is_valid (const char* filename) const
{
  fitsfile* test_fptr = 0;
  int status = 0;

  fits_open_file(&test_fptr, filename, READONLY, &status);
  if (status)
  {
    if (verbose)
    {
      char error[FLEN_ERRMSG];
      fits_get_errstatus(status, error);
      cerr << "FITSFile::is_valid fits_open_file: " << error << endl;
    }
    return false;
  }

  if (verbose)
    cerr << "FITSFile::is_valid test reading MJD" << endl;


  bool result = true;
  try {
    // try to read the MJD from the header
    long day;
    long sec;
    double frac;

    psrfits_read_key (test_fptr, "STT_IMJD", &day);
    psrfits_read_key (test_fptr, "STT_SMJD", &sec);
    psrfits_read_key (test_fptr, "STT_OFFS", &frac);

  }
  catch (Error& error)
  {
    if (verbose)
      cerr << "FITSFile::is_valid failed to read MJD "
        << error.get_message() << endl;
    result = false;
  }

  fits_close_file(test_fptr, &status);
  return result;
}

void read_header(fitsfile* fp, const char* filename, struct fits_params* header)
{
  long day;
  long sec;
  double frac;

  psrfits_read_key(fp, "STT_IMJD", &day);
  psrfits_read_key(fp, "STT_SMJD", &sec);
  psrfits_read_key(fp, "STT_OFFS", &frac);
  header->start_time = MJD((int)day, (int)sec, frac);

  psrfits_move_hdu(fp, "SUBINT");
  psrfits_read_key(fp, "TBIN", &(header->tsamp));
  psrfits_read_key(fp, "NAXIS2", &(header->nrow));
  psrfits_read_key(fp, "NSUBOFFS", &(header->nsuboffs), 0);

  // default is unsigned integers
  psrfits_read_key(fp, "SIGNINT", &(header->signint), 0);
  psrfits_read_key(fp, "ZERO_OFF", &(header->zero_off), 0.0f);
  if (header->zero_off < 0)
    header->zero_off = -header->zero_off;

  /*
  // if unsigned integers used, must have valid zero offset
  if ( (header->signint==0) && (header->zero_off == 0.))
    throw Error(InvalidState, "FITSFile::read_header",
        "Invalid zero offset specified for unsigned data.");
  */

}

void dsp::FITSFile::add_extensions (Extensions* ext)
{
  ext->add_extension (new CloneArchive(archive));
}

void dsp::FITSFile::open_file(const char* filename)
{
  archive = Pulsar::Archive::load(filename);

  unsigned nbits = 0;
  unsigned samples_in_row = 0;
  Reference::To<Pulsar::FITSSUBHdrExtension> ext =
    archive->get<Pulsar::FITSSUBHdrExtension>();

  if (ext)
  {
    nbits = ext->get_nbits();
    samples_in_row = ext->get_nsblk(); // Samples per row.
  }
  else
    throw Error (InvalidState, "FITSFile::open_file",
                 "Could not access FITSSUBHdrExtension");

  const unsigned npol  = archive->get_npol();
  const unsigned nchan = archive->get_nchan();

  int status = 0;
  fits_open_file(&fp, filename, READONLY, &status);
  fits_params header;
  read_header(fp, filename, &header);

  signint = header.signint;
  zero_off = header.zero_off;

  get_info()->set_source(archive->get_source());
  get_info()->set_type(Signal::Pulsar);
  get_info()->set_centre_frequency(archive->get_centre_frequency());
  get_info()->set_bandwidth(archive->get_bandwidth());
  get_info()->set_nchan(nchan);
  get_info()->set_npol(npol);

  if (npol == 1 && archive->get_state() != Signal::Intensity)
  {
    warning << "dsp::FITSFile::open_file npol==1 and data state="
            << archive->get_state() << " (reset to Intensity)" << endl;
    get_info()->set_state( Signal::Intensity );
  }
  else
    get_info()->set_state(archive->get_state());

  get_info()->set_nbit(nbits);
  get_info()->set_rate(1.0/header.tsamp);
  get_info()->set_coordinates(archive->get_coordinates());
  get_info()->set_receiver(archive->get<Pulsar::Receiver>()->get_name());
  get_info()->set_basis(archive->get_basis());

  if (header.nsuboffs < 0)
  {
    throw Error(InvalidState, "FITSFile::open_file",
        "nsuboffs was less than zero");
  }
  get_info()->set_start_time(header.start_time);
  std::string backend_name = archive->get<Pulsar::Backend>()->get_name();
  if (backend_name == "GUPPI" || backend_name == "PUPPI" || backend_name == "VEGAS")
    get_info()->set_machine("GUPPIFITS");
  else if (backend_name == "COBALT")
    get_info()->set_machine("COBALT");
  else
    get_info()->set_machine("FITS");
  get_info()->set_telescope(archive->get_telescope());
  get_info()->set_ndat(header.nrow*samples_in_row);

  set_samples_in_row(samples_in_row);
  set_bytes_per_row((samples_in_row*npol*nchan*nbits) / 8);
  set_number_of_rows(header.nrow);

  data_colnum = dsp::get_colnum(fp, "DATA");
  scl_colnum = dsp::get_colnum(fp, "DAT_SCL");
  offs_colnum = dsp::get_colnum(fp, "DAT_OFFS");
}

void dsp::FITSFile::close ()
{
  // cerr << "dsp::FITSFile::close" << endl;

  int status = 0;

  if (fp)
    fits_close_file (fp, &status);

  fp = NULL;

  if (status)
  {
    fits_report_error (stderr, status);
    throw FITSError (status, "FITSFile::close", "fits_close_file");
  }
}

void dsp::FITSFile::reopen ()
{
  // cerr << "dsp::FITSFile::reopen" << endl;

  if (fp != NULL)
    throw Error (InvalidState, "FITSFile::reopen", "file already open");

  int status = 0;
  fits_open_file (&fp, current_filename.c_str(), READONLY, &status);
  if (status)
    throw FITSError (status, "FITSFile::reopen", "fits_open_file");

  psrfits_move_hdu(fp, "SUBINT");
}

int64_t dsp::FITSFile::seek_bytes (uint64_t bytes)
{ 
  // there should probably be some error checking here ...
  current_byte = bytes;
  return bytes;
}

int64_t dsp::FITSFile::load_bytes(unsigned char* buffer, uint64_t bytes)
{
  // Bytes in a row, within the SUBINT table.
  const unsigned bytes_per_row = get_bytes_per_row();

  // Number of rows in the SUBINT table.
  const unsigned nrow          = get_number_of_rows();

  const unsigned nchan = get_info()->get_nchan();
  const unsigned npol  = get_info()->get_npol();
  const unsigned nbit  = get_info()->get_nbit();
  const unsigned bytes_per_sample = (nchan*npol*nbit) / 8;

  uint64_t bytes_read = 0;

  const unsigned char nval = '0';

  BitSeries* bs = get_output();
  Extension* ext = 0;

  if ( bs->has_extension() )
    ext = dynamic_cast<Extension*>( bs->get_extension() );

  if (!ext)
    bs->set_extension( ext = new Extension );

  ext->zero_off = zero_off;

  // Resize DAT_SCL and DAT_OFFS buffers
  dat_scl.resize(npol*nchan,1);
  dat_offs.resize(npol*nchan,0);

  unsigned irow = 0;

  while (bytes_read < bytes)
  {
    if (verbose)
      cerr << "dsp::FITSFile::load_bytes irow=" << irow 
           << " current_byte=" << current_byte << endl;

    // Calculate the SUBINT table row of the first byte to be read.
    // required_row = [1:nrow]
    unsigned required_row = (current_byte /  bytes_per_row) + 1;

    if (required_row > nrow)
    {
      set_eod(true);
      return bytes_read;
    }

    unsigned byte_offset = current_byte % bytes_per_row;

    // Read from byte_offset to end of the row.
    unsigned this_read = bytes_per_row - byte_offset;

    {
    unsigned bytes_remaining = bytes - bytes_read;
    // Ensure we don't read more than expected.
    if (this_read > bytes_remaining)
      this_read = bytes_remaining;
    }

    if (verbose)
      cerr << "FITSFile::load_bytes row=" << required_row
           << " offset=" << byte_offset << " read=" << this_read << endl;

    int initflag = 0;
    int status = 0;

    // Read the samples
    fits_read_col_byt (fp, data_colnum, required_row, byte_offset+1, 
                       this_read, nval, buffer, &initflag, &status);

    if (status) 
    {
      fits_report_error(stderr, status);
      throw FITSError(status, "FITSFile::load_bytes", "fits_read_col_byt");
    }

    if (required_row != current_row)
    {
      // Read the scales
      fits_read_col(fp,TFLOAT,scl_colnum,required_row,1,nchan*npol,
                    NULL,&(dat_scl[0]),NULL,&status);
      if (status) 
      {
        fits_report_error(stderr, status);
        throw FITSError(status, "FITSFile::load_bytes", "fits_read_col");
      }

      // Read the offsets
      fits_read_col(fp,TFLOAT,offs_colnum,required_row,1,nchan*npol,
                    NULL,&(dat_offs[0]),NULL,&status);
      if (status) 
      {
        fits_report_error(stderr, status);
        throw FITSError(status, "FITSFile::load_bytes", "fits_read_col");
      }

      current_row = required_row;
    }

    if (ext->rows.size() < irow+1)
      ext->rows.resize( irow+1 );

    ext->rows[irow].dat_scl = dat_scl;
    ext->rows[irow].dat_offs = dat_offs;
    ext->rows[irow].nsamp = this_read / bytes_per_sample;

    irow ++;

    buffer += this_read;
    bytes_read += this_read;
    current_byte += this_read;
  }

  return bytes_read;
}

