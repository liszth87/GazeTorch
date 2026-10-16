"""Versioned, local, nonexecutable state. Full numeric exports are opt-in."""
from pathlib import Path
import json, os, tempfile, hashlib
import numpy as np
from calibration import Mapper

DATA=Path.home()/'Library/Application Support/GazeTorch Demo'
SCHEMA=2

def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value

def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(plain(data),indent=2,allow_nan=False)
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def read_json(path):
    def reject(x):raise ValueError('Nonfinite JSON')
    return json.loads(Path(path).read_text(),parse_constant=reject)

def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_profile(path,mapper,config,pose,report,offset):
    atomic_json(path,dict(schema=SCHEMA,type='profile',mapper=mapper.to_dict(),config=config,pose=pose,report=report,offset=offset))

def load_profile(path,config):
    d=read_json(path)
    if d.get('schema')!=SCHEMA or d.get('type')!='profile':raise ValueError('Unsupported profile')
    if d.get('config')!=config or not config.get('camera_id') or not config.get('screen_id') or config.get('framing')!='off':raise ValueError('Setup mismatch or unknown camera/display identity; recalibrate')
    m=Mapper.from_dict(d['mapper'])
    offset=np.asarray(d.get('offset',[0,0]),float)
    if offset.shape!=(2,) or not np.isfinite(offset).all():raise ValueError('Invalid offset')
    pose=d.get('pose')
    if pose is not None and (np.asarray(pose).shape!=(2,6) or not np.isfinite(pose).all() or np.any(np.array(pose[0])>pose[1])):raise ValueError('Invalid pose envelope')
    return m,d

def camera_identity(index):
    """Matches OpenCV 4.11 macOS device ordering. Unknown is explicit."""
    try:
        import AVFoundation as AV
        from functools import cmp_to_key
        devices=list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))+list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
        devices.sort(key=cmp_to_key(lambda a,b:int(a.uniqueID().compare_(b.uniqueID()))))
        device=devices[index]
        framing='unknown'
        if hasattr(device,'isCenterStageActive'):framing='active' if device.isCenterStageActive() else 'off'
        return str(device.uniqueID()),framing
    except Exception:return None,'unknown'
