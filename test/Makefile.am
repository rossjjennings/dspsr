bin_PROGRAMS = test_main bench_InverseFilterbank

test_main_SOURCES = test_main.cpp \
                    test_Apodization.cpp \
                    test_EventEmitter.cpp \
                    test_FIRFilter.cpp \
                    Filterbank/test_FilterbankEngineCPU.cpp \
                    Filterbank/test_FilterbankConfig.cpp \
                    InverseFilterbank/test_InverseFilterbankEngineCPU.cpp \
                    InverseFilterbank/test_InverseFilterbankConfig.cpp \
                    InverseFilterbank/test_InverseFilterbankResponse.cpp \
                    InverseFilterbank/test_InverseFilterbank.cpp \
                    InverseFilterbank/test_InverseFilterbankPipeline.cpp \
                    util.cpp \
                    TransformationProxy.cpp \
                    TestConfig.cpp \
                    InverseFilterbankTestConfig.cpp


if HAVE_CUFFT
  test_main_SOURCES += Filterbank/test_FilterbankEngineCUDA.cpp

  test_main_SOURCES += InverseFilterbank/test_InverseFilterbankEngineCUDA_kernels.cpp \
                       InverseFilterbank/test_InverseFilterbankEngineCUDA.cpp \
                       InverseFilterbank/test_InverseFilterbankEngine_integration.cpp \
                       SpectralKurtosis/test_SpectralKurtosis_integration.cpp

  test_main_SOURCES += test_FTransform_cufft_precision.cpp

  bench_InverseFilterbank_SOURCES = util.cpp \
                                    TestConfig.cpp \
                                    InverseFilterbankTestConfig.cpp \
                                    InverseFilterbank/bench_InverseFilterbank.cpp

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
