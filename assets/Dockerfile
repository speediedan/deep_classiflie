FROM nvidia/cuda:10.2-base-ubuntu18.04
ARG USERNAME
COPY . /tmp/build
RUN ls /tmp/build
VOLUME /experiments/${USERNAME}
# Install some basic utilities and create non-root user
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    unzip \
    bzip2 \
    libx11-6 \
 && apt-get -y autoremove \
 && apt-get clean autoclean \
 && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /var/tmp/* \
 && adduser --disabled-password --gecos '' --shell /bin/bash ${USERNAME} \ 
 && chown "${USERNAME}":"${USERNAME}" /home/${USERNAME} /tmp/build -R \
 && chown "${USERNAME}":"${USERNAME}" /experiments/${USERNAME} -R \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USERNAME}
USER ${USERNAME}
ENV HOME=/home/${USERNAME}
ENV DC_BASE="${HOME}/repos/${USERNAME}" \
    PYTHONPATH="${HOME}/repos/${USERNAME}:${HOME}/repos/captum:${HOME}/repos/transformers" \
    CONDA_AUTO_UPDATE_CONDA=false \
    TARGET_ENV=${USERNAME}
RUN mkdir -p /home/${USERNAME}/repos /home/${USERNAME}/datasets/model_cache/${USERNAME}
# Create a docker volume for the container
WORKDIR /home/${USERNAME}/repos
RUN  git clone https://github.com/pytorch/captum.git \
 && git clone https://github.com/huggingface/transformers \
 && git clone https://github.com/speediedan/deep_classiflie.git
RUN unzip /tmp/build/dc_ds.zip -d /home/${USERNAME}/datasets \
 && unzip /tmp/build/dc_model_alpha.zip -d /home/${USERNAME}/datasets/model_cache/${USERNAME}
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=$HOME/miniconda/bin:$PATH \
    CONDA_DEFAULT_ENV=$TARGET_ENV
RUN conda update -n base -c defaults conda
RUN conda env create -f /tmp/build/deep_classiflie.yml -n ${USERNAME} \
 && conda clean -ya
WORKDIR /home/${USERNAME}/repos/transformers
RUN conda run -n ${TARGET_ENV} pip install . \
 && echo ". $HOME/miniconda/etc/profile.d/conda.sh" >> ~/.profile
# Make RUN commands use the bash shell:
SHELL ["/bin/bash", "-c"]
RUN conda init bash \
 && rm -rf /tmp/build \
 && ls $HOME \
 && env
WORKDIR $DC_BASE
ENTRYPOINT conda run -n $TARGET_ENV python ./deep_classiflie.py
