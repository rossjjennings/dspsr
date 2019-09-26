FROM nexus.engageska-portugal.pt/ska-docker/dspsr-build:latest

ARG MAKE_PROC=1
ARG DSPSR_COMMIT=master

ENV PYTHON=/usr/bin/python

ENV DSPSR_SOURCE=$PSRHOME_SOURCES/dspsr
ENV DSPSR_BUILD=$PSRHOME_BUILD/dspsr
# ENV DSPSR_BUILD=$DSPSR_SOURCE
ENV DSPSR_INSTALL=$PSRHOME_INSTALL/dspsr

RUN git clone https://git.code.sf.net/p/dspsr/code $DSPSR_SOURCE

WORKDIR $DSPSR_SOURCE
RUN git checkout $DSPSR_COMMIT && git pull
RUN ./bootstrap
WORKDIR $DSPSR_BUILD
RUN echo "apsr bpsr cpsr2 caspsr mopsr sigproc dada" > backends.list && \
  $DSPSR_SOURCE/configure --enable-shared --prefix=$DSPSR_INSTALL \
  --with-psrchive-dir=$PSRCHIVE_INSTALL --with-psrxml-dir=$PSRXML_INSTALL \
  --with-psrdada-dir=$PSRDADA_INSTALL \
  --with-cuda-include-dir=$CUDA_INCLUDE_DIR --with-cuda-lib-dir=$CUDA_LIB_DIR
RUN make -j $MAKE_PROC
RUN make install

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DSPSR_INSTALL/lib
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$DSPSR_INSTALL/include
ENV PATH=$PATH:$DSPSR_INSTALL/bin

CMD dspsr --version
