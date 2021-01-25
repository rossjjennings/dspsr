
noinst_LTLIBRARIES = libska1.la

nobase_include_HEADERS = \
  dsp/AAVS2Unpacker.h \
  dsp/CBFPSRUnpacker.h \
  dsp/LFAASPEADUnpacker.h \
  dsp/SKA1Unpacker.h

libska1_la_SOURCES = \
  AAVS2Unpacker.C \
  CBFPSRUnpacker.C \
  LFAASPEADUnpacker.C \
  SKA1Unpacker.C

if HAVE_CUDA

nobase_include_HEADERS += \
  dsp/AAVS2UnpackerCUDA.h \
  dsp/LFAASPEADUnpackerCUDA.h \
  dsp/SKA1UnpackerCUDA.h

libska1_la_SOURCES += \
  AAVS2UnpackerCUDA.cu \
  LFAASPEADUnpackerCUDA.cu \
  SKA1UnpackerCUDA.cu

endif

#############################################################################
#

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CUDA_CFLAGS@
