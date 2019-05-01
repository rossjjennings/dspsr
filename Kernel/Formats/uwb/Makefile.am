
noinst_LTLIBRARIES = libuwb.la

nobase_include_HEADERS =  dsp/UWBUnpacker.h dsp/UWBFloatUnpacker.h dsp/UWBFourBitUnpacker.h dsp/UWBEightBitUnpacker.h

libuwb_la_SOURCES = UWBUnpacker.C UWBFloatUnpacker.C UWBFourBitUnpacker.C UWBEightBitUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/UWBUnpackerCUDA.h dsp/UWBFloatUnpackerCUDA.h
libuwb_la_SOURCES += UWBUnpackerCUDA.cu UWBFloatUnpackerCUDA.cu

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CUDA_CFLAGS@
