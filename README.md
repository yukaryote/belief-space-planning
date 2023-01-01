# belief-space-planning
Implementation of ["Efficient planning in non-Gaussian belief spaces and its application to robot grasping"](https://people.csail.mit.edu/rplatt/papers/platt_isrr2011_6.pdf). 

## Installation
Make a virtualenv with Python 3.8 or newer. Then run 
```
pip install -r requirements.txt
```

This version, as of December 2022, runs on the nightly build of Drake, so you'll have to install that separately [here](https://drake.mit.edu/installation.html).


## Running

Run 
```
python main.py
```

and follow the instructions in the command line to visualize the planned trajectory in MeshCat.
