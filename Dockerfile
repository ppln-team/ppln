FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    unzip \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
    libturbojpeg \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    PyYAML \
    cycler==0.10.0 \
    dill==0.2.8.2 \
    h5py==2.7.1 \
    imgaug==0.2.5 \
    matplotlib==2.2.2 \
    opencv-contrib-python==3.4.2.17 \
    Pillow==5.1.0 \
    scikit-image==0.13.1 \
    scikit-learn==0.19.1 \
    scipy==1.1.0 \
    setuptools==39.1.0 \
    six==1.11.0 \
    tqdm==4.23.4 \
    ipython==7.3.0 \
    ipdb==0.12 \
    albumentations==0.2.2 \
    click==7.0 \
    jpeg4py==0.1.4 \
    addict==2.2.1 \
    colorama==0.4.1

CMD mkdir -p /workspace
WORKDIR /workspace
