FROM condaforge/miniforge3:4.11.0-0 as conda

FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04 AS compile-image

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONDA_DIR=/opt/conda
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib

COPY --from=conda /opt/conda /opt/conda
COPY requirements.txt /tmp/requirements.txt
COPY environment_gpu.yaml /tmp/environment_gpu.yaml

RUN conda env update --prune -f /tmp/environment_gpu.yaml \
    && conda clean -afy \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete

FROM compile-image as cuda-image
ENV PATH=/opt/conda/envs/qdbenchmark38/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH


ENV DISTRO ubuntu2004
ENV CPU_ARCH x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$CPU_ARCH/3bf863cc.pub


COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



WORKDIR $APP_FOLDER
ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng
RUN groupadd --gid ${GROUP_ID} $GROUP && useradd -g $GROUP --uid ${USER_ID} --shell /usr/sbin/nologin -m $USER  && chown -R $USER:$GROUP $APP_FOLDER
USER $USER


FROM cuda-image as dev-image

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    patchelf \
    python3-opengl \
    git \
    python3-dev=3.8* \
    python3-pip &&\
    rm -rf /var/lib/apt/lists/*
RUN pip --no-cache-dir install --no-deps git+https://github.com/adaptive-intelligent-robotics/QDax@qdbenchmark\
    && rm -rf /tmp/*

USER $USER