
SUBDIRS	= config Kernel Signal Management

if HAVE_CXX11
if HAVE_CUDA
SUBDIRS += test
endif
endif

if HAVE_PYTHON
if HAVE_SWIG
SUBDIRS += python
endif
endif

if HAVE_PGPLOT
SUBDIRS += More
endif

include $(top_srcdir)/config/Makefile.root

ACLOCAL_AMFLAGS = -I config

DISTCLEANFILES = backends.list

backends.list:
	cp $(top_srcdir)/config/backends.default backends.list

# make clean deletes the local_include directory (and any out-of-date headers)
clean-local: clean-local-include
.PHONY: clean-local-include
clean-local-include:
	-rm -rf local_include
