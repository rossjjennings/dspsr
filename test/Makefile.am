AM_CPPFLAGS = -I$(top_srcdir)/local_include
LDADD = $(top_srcdir)/Signal/Pulsar/libdspsr.la \
        $(top_srcdir)/Signal/General/libdspdsp.la \
        $(top_srcdir)/Signal/Statistics/libdspstats.la \
        $(top_srcdir)/Kernel/libdspbase.la

bin_PROGRAMS = test_FIRFilter \
               test_InverseFilterbankResponse \
               test_InverseFilterbank \
               test_Apodization \
               test_InverseFilterbankConfig \
               test_InverseFilterbankPipeline

test_FIRFilter_SOURCES = test_FIRFilter.C
test_InverseFilterbankResponse_SOURCES = test_InverseFilterbankResponse.C
test_InverseFilterbank_SOURCES = test_InverseFilterbank.C
test_Apodization_SOURCES = test_Apodization.C
test_InverseFilterbankConfig_SOURCES = test_InverseFilterbankConfig.C
test_InverseFilterbankPipeline_SOURCES = test_InverseFilterbankPipeline.C


if HAVE_CUFFT
  bin_PROGRAMS += test_InverseFilterbankEngineCUDA
  test_InverseFilterbankEngineCUDA_SOURCES = test_InverseFilterbankEngineCUDA.cu
endif


include $(top_srcdir)/config/Makefile.local
