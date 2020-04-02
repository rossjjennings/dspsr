
bin_PROGRAMS = load_bits digihdr

check_PROGRAMS = test_Input test_Unpack

load_bits_SOURCES = load_bits.C
test_Input_SOURCES = test_Input.C
test_Unpack_SOURCES = test_Unpack.C
digihdr_SOURCES = digihdr.C

if HAVE_sigproc
  check_PROGRAMS += sigproc_header
  sigproc_header_SOURCES = sigproc_header.C
endif

#############################################################################


LDADD = $(top_builddir)/Kernel/libdspbase.la @PSRCHIVE_LIBS@ \
	@CUDA_LIBS@ @CUFFT_LIBS@ @OPENMP_CFLAGS@

include $(top_srcdir)/config/Makefile.include

AM_CPPFLAGS += -I$(top_builddir)/local_include 
AM_CPPFLAGS += @CUDA_CFLAGS@ @CUFFT_CFLAGS@ @OPENMP_CFLAGS@

