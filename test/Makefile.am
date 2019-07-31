bin_PROGRAMS = test_main bench_main

bench_main_SOURCES = util.cpp

# test_main_SOURCES = test_main.C \
#                     test_Apodization.C \
#                     test_FIRFilter.C \
#                     test_InverseFilterbankResponse.C \
#                     test_InverseFilterbankConfig.C \
#                     test_InverseFilterbank.C \
#                     test_InverseFilterbankPipeline.C \
#                     test_InverseFilterbankEngineCPU.C \
#                     util.cpp

test_main_SOURCES = test_main.C \
                    test_InverseFilterbankEngineCPU.C \
                    util.cpp


if HAVE_CUFFT

  bench_main_SOURCES += bench_main.cu

  test_main_SOURCES += test_InverseFilterbankEngineCUDA_kernels.cu \
                       test_InverseFilterbankEngineCUDA.cu \
                       test_InverseFilterbankEngine_integration.cu

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
