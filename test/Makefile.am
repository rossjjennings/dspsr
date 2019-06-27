AM_CPPFLAGS = -I$(top_srcdir)/local_include
LDADD = $(top_srcdir)/Signal/Pulsar/libdspsr.la \
        $(top_srcdir)/Signal/General/libdspdsp.la \
        $(top_srcdir)/Signal/Statistics/libdspstats.la \
        $(top_srcdir)/Kernel/libdspbase.la

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
