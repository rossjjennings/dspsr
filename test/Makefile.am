bin_PROGRAMS = test_main

test_main_SOURCES = test_main.C \
                    test_Apodization.C \
                    test_FIRFilter.C \
                    test_InverseFilterbankResponse.C \
                    test_InverseFilterbankConfig.C \
                    test_InverseFilterbank.C \
                    test_InverseFilterbankPipeline.C \
                    util.cpp

if HAVE_CUFFT
  test_main_SOURCES += test_InverseFilterbankEngineCUDA.cu
endif

include $(top_srcdir)/config/Makefile.local
include $(top_srcdir)/config/Makefile.cuda

AM_CPPFLAGS = -I$(top_builddir)/local_include @PSRCHIVE_CPPFLAGS@ @PSRCHIVE_CFLAGS@
LDADD = $(top_builddir)/Signal/Pulsar/libdspsr.la \
        $(top_builddir)/Signal/General/libdspdsp.la \
        $(top_builddir)/Signal/Statistics/libdspstats.la \
        $(top_builddir)/Kernel/libdspbase.la \
				@PSRCHIVE_LIBS@ \
				@CUFFT_LIBS@ @CUDA_LIBS@
