FROM openpcdet-base113:latest

RUN apt-get install libgl1-mesa-glx vim -y

RUN pip install open3d

WORKDIR /home
COPY . .
RUN rm -rf /home/autolabel_pipeline
RUN rm -rf /home/og_git

RUN git config --global --add safe.directory /home


RUN python setup.py develop
RUN pip install numpy==1.21.6

RUN chmod +x /home/tools/eval_loop.sh

ENV PYTHONPATH=/home
ENV XAUTHORITY=/home/.Xauthority
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_GPU_THREAD_COUNT=1

