FROM nvidia/cuda:10.2-base-ubuntu18.04
ARG USERNAME=deep_classiflie
ARG APP_NAME=deep_classiflie
# Install some basic utilities and create non-root user
RUN apt-get update && apt-get install --no-install-recommends -y \
    curl \
    ca-certificates \
    sudo \
    vim \
    git \
    unzip \
    font-manager \
 && apt-get -y autoremove \
 && apt-get clean autoclean \
 && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /var/tmp/* \
 && adduser --disabled-password --gecos '' --shell /bin/bash ${USERNAME} \ 
 && chown -R "${USERNAME}":"${USERNAME}" /home/${USERNAME} \
 && mkdir /experiments \
 && chown -R "${USERNAME}":"${USERNAME}" /experiments \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-${USERNAME}
USER ${USERNAME}
ENV HOME=/home/${USERNAME}
ENV DC_BASE="${HOME}/repos/${APP_NAME}" \
    PYTHONPATH="${HOME}/repos/${USERNAME}:${HOME}/repos/captum:${HOME}/repos/transformers" \
    CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH \
    CONDA_DEFAULT_ENV=deep_classiflie
# Make RUN commands use the bash shell:
SHELL ["/bin/bash", "-c"]
COPY . /tmp/build
RUN mkdir -p /home/${USERNAME}/repos /home/${USERNAME}/datasets/model_cache/${APP_NAME} \
 && git clone https://github.com/pytorch/captum.git /home/${USERNAME}/repos/captum \
 && git clone https://github.com/huggingface/transformers /home/${USERNAME}/repos/transformers \
 && sudo chown -R "${USERNAME}":"${USERNAME}" /tmp/build \
 && mv /tmp/build/${APP_NAME} ${HOME}/repos/ \
 && unzip /tmp/build/dc_ds.zip -d /home/${USERNAME}/datasets \
 && unzip /tmp/build/dc_model_alpha.zip -d /home/${USERNAME}/datasets/model_cache/${APP_NAME} \
 && mv /tmp/build/explore_pred_interpretations.json /home/${USERNAME}/datasets/ \
 && mv /tmp/build/*.pt /home/${USERNAME}/datasets/model_cache/${APP_NAME}/ \
 && mv /tmp/build/*.bin /home/${USERNAME}/datasets/model_cache/${APP_NAME}/ \
 && curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && sudo mv /tmp/build/Humor-Sans-1.0.ttf /usr/local/share/fonts/ \
 && conda update -n base -c defaults conda \
 && conda env create -f /tmp/build/deep_classiflie_docker.yml -n deep_classiflie \
 && conda run -n deep_classiflie pip install /home/${USERNAME}/repos/transformers \
 && conda clean -ya \
 && echo ". $HOME/miniconda/etc/profile.d/conda.sh" >> ~/.profile \
 && sudo find $HOME/miniconda -follow -type f -name '*.a' -delete \
 && sudo find $HOME/miniconda -follow -type f -name '*.pyc' -delete \
 && sudo find $HOME/miniconda -follow -type f -name '*.js.map' -delete \
 && conda init bash \
 && rm -rf /tmp/build
WORKDIR $DC_BASE
CMD ["/bin/bash"]
