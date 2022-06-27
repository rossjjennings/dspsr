
noinst_LTLIBRARIES = libboreas.la

nobase_include_HEADERS = \
  dsp/BoreasSurveyUnpacker.h \
  dsp/BoreasVoltageUnpacker.h

libboreas_la_SOURCES = \
  BoreasSurveyUnpacker.C \
  BoreasVoltageUnpacker.C

if HAVE_CUDA

nobase_include_HEADERS += \
  dsp/BoreasVoltageUnpackerCUDA.h

libboreas_la_SOURCES += \
  BoreasVoltageUnpackerCUDA.cu
  
endif

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

