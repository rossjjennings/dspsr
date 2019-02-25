AM_CPPFLAGS = -I$(top_srcdir)/local_include
LDADD = $(top_srcdir)/Signal/Pulsar/libdspsr.la \
        $(top_srcdir)/Signal/General/libdspdsp.la \
        $(top_srcdir)/Kernel/libdspbase.la

bin_PROGRAMS = test_FIRFilter test_DerippleResponse test_InverseFilterbank

test_FIRFilter_SOURCES = test_FIRFilter.C
test_DerippleResponse_SOURCES = test_DerippleResponse.C
test_InverseFilterbank_SOURCES = test_InverseFilterbank.C

include $(top_srcdir)/config/Makefile.local
