# Unittest
import unittest

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

class TestBBox(unittest.TestCase):

    def test_general(self):

        with self.assertRaises(Exception): b = BBox(1, 1, -3, 0)

        for _ in range(100):
            bbox = BBox.random()

            with self.assertRaises(Exception): bbox.x = "cat"
            with self.assertRaises(Exception): bbox.x = 9
            with self.assertRaises(Exception): b = bbox * -2

    if CYTHON_BBOX_AVAILABLE:
        def test_cython(self):

           for _ in range(100):
              bbox = BBox.random()
              self.assertAlmostEqual(BBox.iou(bbox, bbox), 1)
              self.assertAlmostEqual(BBox.iou(bbox, bbox * 2), 0.25)

              bbox_shift = bbox + (bbox.w, bbox.h)
              self.assertAlmostEqual(BBox.iou(bbox, bbox_shift), 0)

              bbox_shift = bbox + (bbox.w / 2, 0)
              self.assertAlmostEqual(BBox.iou(bbox, bbox_shift), 1/3)

    def test_operators(self):

        for _ in range(100):
            bbox = BBox.random()

            self.assertAlmostEqual(bbox + 7, BBox(bbox.x + 7, bbox.y + 7, bbox.w, bbox.h))
            self.assertAlmostEqual(bbox + (-3, -71), BBox(bbox.x - 3, bbox.y - 71, bbox.w, bbox.h))

            self.assertAlmostEqual((bbox * 3).area,  bbox.area * 9)

if __name__ == "__main__":
    unittest.main()