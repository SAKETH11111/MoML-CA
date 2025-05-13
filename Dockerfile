# Dockerfile
ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04
# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Set environment variables to ensure UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create and activate the Conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -all -f -y

# Make Conda environment activate by default in new shells
SHELL ["conda", "run", "-n", "moml_ca_env", "/bin/bash", "-c"]

# Copy the rest of the application code
WORKDIR /app
COPY . /app

# Install any remaining pip dependencies from requirements.txt
# (Useful if some packages are not on Conda or need specific versions via pip)
# Ensure this runs *within* the activated Conda environment
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Set the default command for the container (optional)
# CMD ["python", "your_main_script.py"]

# Expose ports if necessary (e.g., for Jupyter or a web UI)
# EXPOSE 8888

# Verify environment (optional, good for debugging)
RUN echo "MoML-CA Environment Setup Complete"
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
RUN python -c "import rdkit; print(f'RDKit version: {rdkit.__version__}')"
RUN python -c "import openmm; print(f'OpenMM version: {openmm.version.version}')"