FROM tensorflow/tensorflow:latest-gpu
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y python-msgpack
RUN apt-get install -y python-zmq
RUN apt-get install -y python-pil
RUN apt-get install -y python-sqlite
RUN apt-get install -y sqlite3
RUN apt-get install -y python-h5py

RUN apt-get install -y git
RUN apt-get install -y python-setuptools

RUN cd /root && git clone https://github.com/tmbdev/tfndlstm.git && \
        cd tfndlstm && python setup.py install && true && true
RUN cd /root && git clone https://github.com/tmbdev/tfspecs.git && \
        cd tfspecs && python setup.py install && true && true

# move to cupy
RUN apt-get install -y libyaml-dev
RUN pip install PyYAML

