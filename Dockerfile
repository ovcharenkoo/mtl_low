FROM nvcr.io/nvidia/pytorch:22.03-py3

ARG MYUSER=this_user
ARG MYUID=1000
ARG MYGID=1000

ENV DEBIAN_FRONTEND noninteractive

# Install cmake, add user
RUN mkdir -p /workspace && cd /workspace && \
    if [ -f /usr/bin/yum ]; then yum install -y make wget vim libfftw3-dev scons python3-distutils man; fi && \
    if [ -f /usr/bin/apt-get ]; then apt-get update && apt-get install -y apt-utils make man wget vim libfftw3-dev scons python3-distutils; fi && \
    groupadd -f -g ${MYGID} ${MYUSER} && \
    useradd -rm -u $MYUID -g $MYUSER -p "" $MYUSER && \
    chown ${MYUSER}:${MYGID} /workspace 

USER $MYUSER

RUN pip install --no-cache-dir natsort ipynb segyio scikit-image latex seaborn

RUN echo "#!/bin/bash" >> /workspace/install_denise.sh && \
    echo "cd /workspace/project/" >> /workspace/install_denise.sh && \
    echo "wget https://github.com/daniel-koehn/DENISE-Black-Edition/archive/master.zip -O DENISE-Black-Edition.zip" >> /workspace/install_denise.sh && \
    echo "unzip DENISE-Black-Edition.zip && rm DENISE-Black-Edition.zip" >> /workspace/install_denise.sh && \
    echo "cd /workspace/project/DENISE-Black-Edition-master/libcseife" >> /workspace/install_denise.sh && \
    echo "make" >> /workspace/install_denise.sh && \
    echo "cd /workspace/project/DENISE-Black-Edition-master/src" >> /workspace/install_denise.sh && \
    echo "make -j 4 denise" >> /workspace/install_denise.sh && \
    chmod 755 /workspace/install_denise.sh

RUN echo "#!/bin/bash" >> /workspace/install_madagascar.sh && \
    echo "cd /workspace/project/" >> /workspace/install_madagascar.sh && \
    echo "git clone https://github.com/ahay/src RSFSRC" >> /workspace/install_madagascar.sh && \
    echo "cd /workspace/project/RSFRC" >> /workspace/install_madagascar.sh && \
    echo "./configure --prefix=/workspace/project/madagascar" >> /workspace/install_madagascar.sh && \
    echo "make -j 4 install" >> /workspace/install_madagascar.sh && \
    echo "cd /workspace/project/" >> /workspace/install_madagascar.sh && \
    echo "source /workspace/project/madagascar/share/madagascar/etc/env.sh" >> /workspace/install_madagascar.sh && \
    chmod 755 /workspace/install_madagascar.sh

# RUN echo "source /workspace/project/madagascar/share/madagascar/etc/env.sh" >> /${MYUSER}/.bashrc

WORKDIR /workspace/project

