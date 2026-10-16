"""Regression for native NumPy stack overflow in the real camera QThread.

Uses a fixture camera, but real face detection, normalization, and model inference.
Run: .venv/bin/python demo/test_camera_worker.py
"""
import unittest
import os
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, Qt
from app import CameraWorker, ROOT


class CameraWorkerTest(unittest.TestCase):
    def test_face_inference_and_repeated_shutdown(self):
        application = QCoreApplication.instance() or QCoreApplication([])
        path = os.environ.get('GAZETORCH_TEST_FACE', str(ROOT / 'demo/test-assets/face.jpg'))
        frame = cv2.imread(path)
        self.assertIsNotNone(frame, 'Provide a consented face image via GAZETORCH_TEST_FACE or demo/test-assets/face.jpg')
        for cycle in range(3):
            worker = CameraWorker(0)
            samples, statuses = [], []

            class FixtureCamera:
                released = False
                def set(self, *args): return True
                def isOpened(self): return True
                def read(self): return True, frame.copy()
                def release(self): self.released = True

            camera = FixtureCamera()
            def receive(features, preview, milliseconds):
                samples.append(features)
                if len(samples) >= 10:
                    worker.requestInterruption()

            worker.sample.connect(receive, Qt.ConnectionType.DirectConnection)
            worker.status.connect(statuses.append, Qt.ConnectionType.DirectConnection)
            with patch('app.cv2.VideoCapture', return_value=camera):
                worker.start()
                completed = worker.wait(60000)
                if not completed:
                    worker.requestInterruption()
                    worker.wait()
                self.assertTrue(completed, 'Worker timed out')
            self.assertEqual(statuses, ['Loading GazeTorch…'])
            self.assertEqual(len(samples), 10)
            self.assertTrue(all(s is not None and np.isfinite(s).all() for s in samples))
            self.assertTrue(camera.released)
            print(f'Cycle {cycle + 1}: 10 valid gaze predictions; camera released', flush=True)


if __name__ == '__main__':
    unittest.main()
