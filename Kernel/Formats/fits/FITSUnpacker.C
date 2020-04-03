/***************************************************************************
 *
 *   Copyright (C) 2008 by Jonathan Khoo & Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/FITSUnpacker.h"
#include "dsp/FITSFile.h"
#include "Error.h"

#define ONEBIT_MASK 0x1
#define TWOBIT_MASK 0x3
#define FOURBIT_MASK 0xf

typedef float (dsp::FITSUnpacker::*BitNumberFn) (const int num);

using std::cerr;
using std::endl;
using std::vector;
using std::cout;

// Default zero offset for one-bit data.
const float ONEBIT_SCALE = 0.5;

// Default zero offset for four-bit data.
const float FOURBIT_SCALE = 7.5;

// Default zero offset for eight-bit data.
const float EIGHTBIT_SCALE = 31.5;

// Number of bits per byte.
const int BYTE_SIZE = 8;

dsp::FITSUnpacker::FITSUnpacker(const char* name) : Unpacker(name)
{
}

/**
 * @brief Iterate each row (subint) and sample extracting the values
 *        from input buffer and placing the scaled value in the appropriate
 *        position address by 'into'.
 * @throws InvalidState if nbit != 1, 2, 4 or 8.
 */

void dsp::FITSUnpacker::unpack()
{

  // Allocate mapping method to use depending on how many bits per value.
  BitNumberFn p;
  const unsigned nbit = input->get_nbit();

  if (verbose)
    cerr << "dsp::FITSUnpacker::unpack with nbit=" << nbit << endl;

  switch (nbit) {
    case 1:
      p = &dsp::FITSUnpacker::oneBitNumber;
      break;
    case 2:
      p = &dsp::FITSUnpacker::twoBitNumber;
      break;
    case 4:
      p = &dsp::FITSUnpacker::fourBitNumber;
      break;
    case 8:
      p = &dsp::FITSUnpacker::eightBitNumber;
      break;
    case 16:
      break;
    default:
      throw Error(InvalidState, "FITSUnpacker::unpack",
          "invalid nbit=%d", nbit);
  }

  const unsigned npol  = input->get_npol();
  const unsigned nchan = input->get_nchan();
  const unsigned ndat  = input->get_ndat();

  // Number of samples in one byte.
  const int samples_per_byte = BYTE_SIZE / nbit;
  const int mod_offset = samples_per_byte - 1;

  const unsigned char* from = input->get_rawptr();
  const int16_t* from16 = (int16_t *)input->get_rawptr();

  const FITSFile::Extension* ext = dynamic_cast<const FITSFile::Extension*>( get_input()->get_extension() );

  zero_off = ext->zero_off;

  // more optimal 8-bit unpacker
  if (nbit == 8)
  {
    const float* scl = &(ext->dat_scl[0]);
    const float* off = &(ext->dat_offs[0]);

    const unsigned nchanpol = nchan * npol;
    for (unsigned ipol = 0; ipol < npol; ++ipol)
    {
#ifdef HAVE_OPENMP
      #pragma omp parallel for
#endif
      for (unsigned ichan = 0; ichan < nchan; ++ichan)
      {
        const unsigned ipolchan = ipol * nchan + ichan;

        // unpacking from TPF order to FPT
        float* into = output->get_datptr(ichan, ipol);
        const unsigned char * from_ptr = from + ipolchan;

        // samples stored in TPF order
        for (unsigned idat = 0; idat < ndat; ++idat)
        {
          into[idat] = ((*this.*p)(*from_ptr)) * scl[ipolchan] + off[ipolchan];
          from_ptr += nchanpol;
        }
      }
    }
  }
  else if (nbit<8)
  {

    // Iterate through input data, split the byte depending on number of
    // samples per byte, get corresponding mapped value and store it
    // as pol-chan-dat.
    //
    // TODO: Use a lookup table???
    for (unsigned idat = 0; idat < ndat; ++idat) {
      const float* scl = &(ext->dat_scl[0]);
      const float* off = &(ext->dat_offs[0]);
      for (unsigned ipol = 0; ipol < npol; ++ipol) {
        for (unsigned ichan = 0; ichan < nchan;) {

          const int mod = mod_offset - (ichan % samples_per_byte);
          const int shifted_number = *from >> (mod * nbit);

          float* into = output->get_datptr(ichan, ipol) + idat;

#if 0
          cerr << "ipol=" << ipol << " ichan=" << ichan 
               << " scl=" << *scl << " off=" << *off << endl;
#endif
   
          *into = (*this.*p)(shifted_number) * (*scl) + (*off);
          ++scl; ++off;

          // Move to next byte when the entire byte has been split.
          if ((++ichan) % (samples_per_byte) == 0) {
            ++from;
          }
        }
      }
    }
  }

  else if (nbit==16) {
    for (unsigned idat=0; idat<ndat; idat++) {
      const float* scl = &(ext->dat_scl[0]);
      const float* off = &(ext->dat_offs[0]);
      for (unsigned ipol=0; ipol<npol; ipol++) {
        for (unsigned ichan=0; ichan<nchan; ichan++) {
          float* into = output->get_datptr(ichan, ipol) + idat;
          *into = (float)(*from16) * (*scl) + (*off);
          ++scl; ++off;
          ++from16;
        }
      }
    }
  }

}

bool dsp::FITSUnpacker::matches(const Observation* observation)
{
  return observation->get_machine() == "FITS";
}


/**
 * @brief Mask and scale ( - 0.5) a one-bit number.
 * @param int Contains an unsigned one-bit number to be masked and scaled
 * @return Scaled one-bit value.
 */

float dsp::FITSUnpacker::oneBitNumber(const int num)
{
  const int masked_number = num & ONEBIT_MASK;
  //return masked_number - ONEBIT_SCALE;
  return masked_number - zero_off;
}


/**
 * @brief Scale the eight-bit number ( - 31.5).
 * @param int Eight-bit number to be scaled
 * @return Scaled eight-bit value.
 */

float dsp::FITSUnpacker::eightBitNumber(const int num)
{
  //return num - EIGHTBIT_SCALE;
  return num - zero_off;
}


/**
 * @brief Mask (0xf) and scale ( - 7.5) an unsigned four-bit number.
 * @param int Contains the unsigned four-bit number to be scaled.
 * @returns float Scaled four-bit value.
 */

float dsp::FITSUnpacker::fourBitNumber(const int num)
{
  const int masked_number = num & FOURBIT_MASK;
  //return masked_number - FOURBIT_SCALE;
  return masked_number - zero_off;
}


/**
 * @brief Mask (0x3) and scale an unsigned two-bit number:
 *        0 = -2
 *        1 = -0.5
 *        2 = 0.5
 *        3 = 2.0
 *
 * @param int Contains the unsigned two-bit number to be scaled.
 * @returns float Scaled two-bit value.
 */

float dsp::FITSUnpacker::twoBitNumber(const int num)
{
  const int masked_number = num & TWOBIT_MASK;
  switch (masked_number) {
    case 0:
      return -2.0;
    case 1:
      return -0.5;
    case 2:
      return 0.5;
    case 3:
      return 2.0;
    default:
      return 0.0;
  }
}
