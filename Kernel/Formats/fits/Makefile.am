noinst_LTLIBRARIES = libfits.la

nobase_include_HEADERS = dsp/FITSUnpacker.h dsp/FITSFile.h \
			 dsp/GUPPIFITSUnpacker.h dsp/FITSDigitizer.h \
             dsp/FITSOutputFile.h

libfits_la_SOURCES = FITSUnpacker.C GUPPIFITSUnpacker.C \
		     FITSFile.C fits_params.h FITSDigitizer.C \
             FITSOutputFile.C

if HAVE_CUDA

nobase_include_HEADERS += dsp/FITSDigitizerCUDA.h
libfits_la_SOURCES += FITSDigitizerCUDA.cu

endif


#############################################################################
#

libfits_la_LIBADD = @CFITSIO_LIBS@ @OPENMP_CFLAGS@ @CUDA_LIBS@

include $(top_srcdir)/config/Makefile.include
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS += @CFITSIO_CFLAGS@
AM_CXXFLAGS = @OPENMP_CFLAGS@ @CUDA_CFLAGS@

