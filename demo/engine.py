"""Local webcam adapter for upstream GazeTorch's MobileNet checkpoint."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
import mediapipe as mp
from cv.normalization import normalize

ASSETS = ROOT / 'demo' / 'assets'

def normalization_origin(face,rv,tv):
    rotation,_=cv2.Rodrigues(rv)
    world=(rotation@face[[33,133,362,263,98,327]].T+np.asarray(tv).reshape(3,1)).T
    return (world[:4].mean(0)+world[4:].mean(0))/2

class GazeEngine:
    def __init__(self, camera_profile=None):
        self.camera_profile=camera_profile
        self.details={}
        self.maps=None
        torch.set_num_threads(2)
        # Official author checkpoint contains the upstream Python model, not just weights.
        self.model = torch.load(ASSETS / 'GazeTorch_MobileNet.pkl', map_location='cpu', weights_only=False).eval()
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=.6, min_tracking_confidence=.6)
        points = [list(map(float, line.split()[1:4])) for line in
                  (ASSETS / 'canonical_face_model.obj').read_text().splitlines() if line.startswith('v ')]
        self.face = np.array(points, dtype=np.float64) * [10, -10, -10] # cm -> mm, camera axes
        self.ids = [33, 133, 362, 263, 98, 327, 1, 4, 152, 61, 291, 168]

    def infer(self, frame):
        self.details={'reason':'face not detected'}
        h,w=frame.shape[:2]
        camera=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=float)
        if self.camera_profile is not None:
            if list(self.camera_profile['frame_size'])!=[w,h]:
                raise ValueError('Camera profile resolution changed')
            camera=np.array(self.camera_profile['K_new'],float)
            if self.maps is None:
                self.maps=cv2.initUndistortRectifyMap(np.array(self.camera_profile['K']),np.array(self.camera_profile['distortion']),None,camera,(w,h),cv2.CV_32FC1)
            frame=cv2.remap(frame,*self.maps,cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        pts = np.array([[p.x, p.y] for p in result.multi_face_landmarks[0].landmark])
        pixels=pts*[w,h]
        def eyelids(points):
            widths=[np.linalg.norm(points[33]-points[133]),np.linalg.norm(points[362]-points[263])]
            return [np.linalg.norm(points[159]-points[145])/max(widths[0],1e-8),np.linalg.norm(points[386]-points[374])/max(widths[1],1e-8)]
        openness=eyelids(pixels);legacy_open=eyelids(pts)
        self.details={'reason':'eyes closed','openness':tuple(openness)}
        # Conservative floor; session supplies personal reference after preflight.
        if min(openness)<.06 and min(legacy_open)<.10:return None
        ok, rv, tv = cv2.solvePnP(self.face[self.ids], pixels[self.ids], camera,
                                 np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
        if not ok or tv[2,0] <= 0:
            return None
        crop, rotation, _ = normalize(rgb, camera, self.face, rv, tv,
                                     id_calib=(33,133,362,263,98,327))
        # Matches upstream config/augmenter/default_test.yaml: per-image/channel standardization.
        x = crop.astype(np.float32)
        x = (x-x.mean(axis=(0,1))) / np.maximum(x.std(axis=(0,1)), 1.)
        with torch.inference_mode():
            prediction = self.model([torch.from_numpy(x.transpose(2,0,1).copy())[None]])['head'][0][0].numpy()
        pitch, yaw = prediction
        direction = np.array([-np.cos(pitch)*np.sin(yaw), -np.sin(pitch), -np.cos(pitch)*np.cos(yaw)])
        direction = rotation.T @ direction
        if abs(direction[2]) < .1:
            return None
        features = np.array([direction[0]/-direction[2], direction[1]/-direction[2],
                             tv[0,0]/tv[2,0], tv[1,0]/tv[2,0]])
        if not np.isfinite(features).all():
            return None
        origin=normalization_origin(self.face,rv,tv)
        gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
        self.details=dict(reason='' if min(openness)>=.06 else 'eyes closed',openness=tuple(openness),
            origin=tuple(origin),direction=tuple(direction),pose=tuple(np.r_[rv.ravel(),tv.ravel()]),
            quality=(float(gray.mean()),float(cv2.Laplacian(gray,cv2.CV_64F).var())),
            legacy_features=tuple(features) if min(legacy_open)>=.10 else ())
        return features, crop

    def close(self):
        self.mesh.close()

class Calibration:
    def fit(self, features, targets):
        x = np.asarray(features); y = np.asarray(targets)
        if len(x) < 9 or not np.isfinite(x).all():
            raise ValueError('Not enough valid calibration samples.')
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), .005)
        z = (x-self.mean)/self.scale
        a = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(a.shape[1]) * .3; penalty[0,0] = 0
        self.coef = np.linalg.solve(a.T@a + penalty, a.T@y)
        return self

    def predict(self, features):
        z = (np.asarray(features)-self.mean)/self.scale
        return np.r_[1.,z] @ self.coef

if __name__ == '__main__':
    engine = GazeEngine()
    with torch.inference_mode():
        a = engine.model([torch.zeros(1,3,224,224)])['head'][0]
        b = engine.model([torch.randn(1,3,224,224)])['head'][0]
    assert a.shape == (1,2) and torch.isfinite(a).all() and not torch.allclose(a,b)
    # Synthetic held-out check: a known mapping must generalize, not just fit anchors.
    rng = np.random.default_rng(42); x=rng.normal(size=(100,4)); y=x[:,:2]*.1+.5
    cal=Calibration().fit(x[:80],y[:80])
    err=np.max(np.abs(np.array([cal.predict(v) for v in x[80:]])-y[80:]))
    assert err < .01, err
    engine.close()
    print('PASS: actual checkpoint inference, landmark engine initialization, held-out calibration mapping')
