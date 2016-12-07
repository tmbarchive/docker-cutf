# TensorFlow - Docker

This is a simple TensorFlow docker container. It's based on (and similar to)
the original Google TensorFlow docker container but adds some new packages
and comes with some scripts that make working with TensorFlow a little easier.

To build the container, just type `make`.

To run the container, run `./run`. This will start up an iPython notebook
server on port 8888 (if you want a different port, use `port=9999 ./run`).
The directory ./data is mapped to /data, and the directory ./models is
mapped to /models (both can by symlinks). The current directory is 
mapped to /notebooks.

To run the command line script, use `./run python mnist-specs.py`.
