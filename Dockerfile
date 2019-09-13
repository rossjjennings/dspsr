FROM nexus.engageska-portugal.pt/ska-docker/dspsr-build:latest

ARG MAKE_PROC=1
ARG DSPSR_COMMIT=master

ENV PYTHON=/usr/bin/python

ENV DSPSR=$PSRHOME/dspsr
# ENV DSPSR_BUILD=$HOME/build/dspsr
ENV DSPSR_BUILD=$DSPSR
ENV DSPSR_INSTALL=$HOME/software/dspsr

RUN git clone https://git.code.sf.net/p/dspsr/code $DSPSR

WORKDIR $DSPSR
RUN git checkout $DSPSR_COMMIT && git pull 
RUN ./bootstrap
WORKDIR $DSPSR_BUILD
RUN echo "apsr bpsr cpsr2 caspsr mopsr sigproc dada" > backends.list && $DSPSR/configure --enable-shared --prefix=$DSPSR_INSTALL --with-psrchive-dir=$PSRCHIVE_INSTALL --with-psrxml-dir=$PSRXML_INSTALL --with-psrdada-dir=$PSRDADA_INSTALL
RUN make -j $MAKE_PROC
RUN make install 

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DSPSR_INSTALL/lib
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$DSPSR_INSTALL/include
ENV PATH=$PATH:$DSPSR_INSTALL/bin

CMD dspsr --version

