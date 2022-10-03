FROM nvidia/cuda:11.3.0-devel-ubuntu18.04
ARG CMAKE_VERSION
ARG LLVM_VERSION

COPY requirements.txt /tmp/requirements.txt

# RUN apt update -y
# RUN apt install -y software-properties-common
# RUN apt-add-repository -y ppa:graphics-drivers/ppa

RUN echo "deb https://repo.scala-sbt.org/scalasbt/debian all main" > /etc/apt/sources.list.d/sbt.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823

# LLVM repository with the version we are building
# used to install clang-format without building LLVM, for style checks
#RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main" >> /etc/apt/sources.list.d/llvm.list
RUN apt-key adv --fetch-keys https://apt.llvm.org/llvm-snapshot.gpg.key

RUN apt update  -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends $(sed s/#.*$//g tmp/requirements.txt) && \
    apt purge -y cmake && \
    apt autoremove -y && \
    apt autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log

RUN ldconfig /usr/local/cuda/lib64

RUN curl -O https://repo1.maven.org/maven2/org/apache/calcite/avatica/avatica/1.13.0/avatica-1.13.0.jar

COPY ccache.patch /tmp/ccache.patch
RUN curl -L -O https://github.com/ccache/ccache/releases/download/v3.7.12/ccache-3.7.12.tar.gz && \
    tar xzf ccache-3.7.12.tar.gz && \
    rm ccache-3.7.12.tar.gz && \
    cd ccache-3.7.12 && \
    cd src && patch language.c /tmp/ccache.patch && cd .. && \
    ./configure && \
    make && \
    make install

COPY LLVM-${LLVM_VERSION}-Linux.deb /tmp/LLVM-${LLVM_VERSION}-Linux.deb

RUN dpkg -i /tmp/LLVM-${LLVM_VERSION}-Linux.deb && \
    rm /tmp/LLVM-${LLVM_VERSION}-Linux.deb

COPY cmake-${CMAKE_VERSION}-Linux-x86_64.deb /tmp/cmake-${CMAKE_VERSION}-Linux-x86_64.deb

RUN dpkg -i /tmp/cmake-${CMAKE_VERSION}-Linux-x86_64.deb && \
    rm /tmp/cmake-${CMAKE_VERSION}-Linux-x86_64.deb

RUN apt update  -y && \
    # apt upgrade -y && \
    apt install -y --no-install-recommends libpostgresql-jdbc-java && \
    apt autoremove -y && \
    apt autoclean

RUN pip install --user wheel && pip install --user JayDeBeApi JPype1==0.6.3

ENV CCACHE_COMPILERCHECK=content
