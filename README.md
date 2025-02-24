# alain-utils
Various Python utilities including loggers, SE(3) poses, 2D bounding boxes, function decorators, kalman filter, Plotly functions etc.

## Installation
Clone and install the package via `pip`:
```sh
git clone git@github.com:AlainSchoebi/alain-utils.git
pip install ./alain-utils
```

To install the package in editable mode for development:
```sh
git clone git@github.com:AlainSchoebi/alain-utils.git
pip install -e ./alain-utils
```

## Additional packages
Some utilities might only be available when additional packages are installed, such as matplotlib, pycolmap, cv2, tqdm, rospy, plotly, shapely and cython-bbox.

## Tests
To run tests:
```sh
python -m pytest
```
Note that simply running `pytest` might not work and lead to `ImportError`.