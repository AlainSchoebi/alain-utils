# Pytest
import pytest

# Numpy
import numpy as np

# Cython BBox
try:
    import cython_bbox
    CYTHON_BBOX_AVAILABLE = True
except ImportError:
    CYTHON_BBOX_AVAILABLE = False
    pass

# Utils
from alutils import BBox

# Logging
from alutils import get_logger
get_logger("alutils").setLevel("CRITICAL")

def test_general():

    with pytest.raises(Exception): b = BBox(1, 1, -3, 0)

    for _ in range(100):
        bbox = BBox.random()

        with pytest.raises(Exception): bbox.x = "cat"
        with pytest.raises(Exception): bbox.x = 9
        with pytest.raises(Exception): b = bbox * -2

if CYTHON_BBOX_AVAILABLE:
    def test_cython():

        for _ in range(100):
            bbox = BBox.random()
            assert BBox.iou(bbox, bbox) == pytest.approx(1)
            assert BBox.iou(bbox, bbox * 2) == pytest.approx(0.25)

            bbox_shift = bbox + (bbox.w, bbox.h)
            assert BBox.iou(bbox, bbox_shift) == pytest.approx(0)

            bbox_shift = bbox + (bbox.w / 2, 0)
            assert BBox.iou(bbox, bbox_shift) == pytest.approx(1/3)

def test_operators():

    for _ in range(100):
        bbox = BBox.random()

        assert bbox + 7 == BBox(bbox.x + 7, bbox.y + 7, bbox.w, bbox.h)
        assert bbox + (-3, -71) == BBox(bbox.x - 3, bbox.y - 71, bbox.w, bbox.h)
        assert (bbox * 3).area == pytest.approx(bbox.area * 3**2)