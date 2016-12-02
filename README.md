# ISS0031 Homework

Here can be found codes used for the homework report for "ISS031 Modelling and Identification" course.

## Files

### Numpy implementation

Running the example.

```bash
$ ipython -i train.py

# or
$ python -i train.py
```

Related files:
- Training the network: train.py
- BaseNetwork class: base_network.py
- FeedForward class: feed_forward.py

Dependencies:
- Numpy

### PyBrain

Running the example.

```bash
$ ipython -i train_pb.py
```

Related files:
- Training the network: train_pb.py
- BaseNetwork class: feed_forward_pybrain.py

Dependencies:
- Numpy
- Pybrain

### Keras

```bash
$ ipython -i train_keras.py
```

Related files:
- Training the network: train_keras.py
- BaseNetwork class: feed_forward_keras.py

Dependencies:
- Numpy
- Pybrain
- Keras
- TensorFlow or Theano

### Other

- Loading data files: minst_data.py
- Helper functions: tools.py
- Other modules: pickle, gzip, matplotlib
