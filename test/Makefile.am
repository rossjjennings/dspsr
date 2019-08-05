bin_PROGRAMS = test_main

test_main_SOURCES = test_main.cpp \
                    test_Apodization.cpp \
                    test_FIRFilter.cpp \
                    InverseFilterbank/test_InverseFilterbankResponse.cpp \
                    InverseFilterbank/test_InverseFilterbankConfig.cpp \
                    InverseFilterbank/test_InverseFilterbank.cpp \
                    InverseFilterbank/test_InverseFilterbankPipeline.cpp \
                    InverseFilterbankTestConfig.cpp \
                    util.cpp

                    # InverseFilterbank/test_InverseFilterbankEngineCPU.cpp

# test_main_SOURCES = test_main.cpp \
#                     InverseFilterbank/test_InverseFilterbankEngineCPU.cpp \
#                     util.cpp


if HAVE_CUFFT
  # bin_PROGRAMS += bench_InverseFilterbankEngineCUDA
  # bench_InverseFilterbankEngineCUDA_SOURCES = InverseFilterbank/bench_InverseFilterbankEngineCUDA.cu

  # test_main_SOURCES += InverseFilterbank/test_InverseFilterbankEngineCUDA_kernels.cu

  # test_main_SOURCES += InverseFilterbank/test_InverseFilterbankEngineCUDA_kernels.cu \
  #                      InverseFilterbank/test_InverseFilterbankEngineCUDA.cu

  test_main_SOURCES += InverseFilterbank/test_InverseFilterbankEngineCUDA_kernels.cu \
                       InverseFilterbank/test_InverseFilterbankEngineCUDA.cu \
                       InverseFilterbank/test_InverseFilterbankEngine_integration.cu

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
