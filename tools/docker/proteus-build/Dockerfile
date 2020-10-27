FROM nvidia/cuda:11.1-devel-ubuntu18.04

COPY requirements.txt /tmp/requirements.txt

# RUN apt update -y
# RUN apt install -y software-properties-common
# RUN apt-add-repository -y ppa:graphics-drivers/ppa

RUN echo "deb https://dl.bintray.com/sbt/debian /" >> /etc/apt/sources.list.d/sbt.list
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823

# LLVM repository with the version we are building
# used to install clang-format without building LLVM, for style checks
RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main" >> /etc/apt/sources.list.d/llvm.list
RUN apt-key adv --fetch-keys https://apt.llvm.org/llvm-snapshot.gpg.key

RUN apt update  -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends $(sed s/#.*$//g tmp/requirements.txt) && \
    apt autoremove -y && \
    apt autoclean && \
    rm -rf /var/lib/apt/lists/* /var/log/dpkg.log && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-11 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-11 100 && \
    update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-11 100 && \
    update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-11 100

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

RUN git clone https://gitlab.kitware.com/cmake/cmake.git && \
    cd cmake && \
    git checkout e647949539f4704a00eddac2357d59ceeb8bc0ca && \
    ./bootstrap --parallel=$(nproc --all) -- -DCMAKE_USE_OPENSSL=OFF && \
    make -j$(nproc --all) && \
    make install && \
    cd .. && \
    rm -rf cmake


RUN apt update -y && \
    apt install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt purge -y software-properties-common

RUN apt update  -y && \
    apt upgrade -y && \
    apt install -y --no-install-recommends libpostgresql-jdbc-java libstdc++-10-dev && \
    apt autoremove -y && \
    apt autoclean

RUN pip install --user wheel && pip install --user JayDeBeApi JPype1==0.6.3

ENV CCACHE_COMPILERCHECK=content
