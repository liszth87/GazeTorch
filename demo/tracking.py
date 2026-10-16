"""Posture, temporal filtering, and conservative translation checks."""
import numpy as np

class OneEuro:
    def __init__(self,min_cutoff=.7,beta=.002,derivative_cutoff=1.):
        self.min_cutoff=min_cutoff;self.beta=beta;self.derivative_cutoff=derivative_cutoff;self.reset()
    def reset(self):self.time=None;self.value=None;self.raw=None;self.derivative=np.zeros(2)
    def apply(self,value,now):
        value=np.asarray(value,float)
        if not np.isfinite(value).all():self.reset();return None
        if self.time is not None and now<=self.time:return None
        if self.time is None or now-self.time>.25:
            self.time=now;self.raw=value.copy();self.value=value.copy();self.derivative=np.zeros(2);return value.copy()
        dt=now-self.time
        def alpha(c):return 1/(1+1/(2*np.pi*c*dt))
        a=alpha(self.derivative_cutoff)
        self.derivative=a*(value-self.raw)/dt+(1-a)*self.derivative
        a=alpha(self.min_cutoff+self.beta*np.linalg.norm(self.derivative))
        self.value=a*value+(1-a)*self.value
        self.raw=value.copy();self.time=now;return self.value.copy()

class PoseGuard:
    def __init__(self,poses=None,limits=None):
        if limits is not None:self.low=np.array(limits[0]);self.high=np.array(limits[1])
        elif poses:
            p=np.array(poses);margin=np.array([.10,.10,.10,30,30,50])
            self.low=np.min(p,axis=0)-margin;self.high=np.max(p,axis=0)+margin
        else:self.low=None;self.high=None
        self.departed=None;self.returned=None;self.blocked=False
    def update(self,pose,now):
        if self.low is None:return True
        if len(pose)!=6 or not np.isfinite(pose).all():self.blocked=True;return False
        inside=np.all(np.array(pose)>=self.low)&np.all(np.array(pose)<=self.high)
        if not inside:
            self.returned=None
            if self.departed is None:self.departed=now
            if now-self.departed>=.3:self.blocked=True
        else:
            self.departed=None
            if self.returned is None:self.returned=now
            if now-self.returned>=.35:self.blocked=False
        return not self.blocked
    def to_list(self):return [self.low.tolist(),self.high.tolist()] if self.low is not None else None

class Tracker:
    def __init__(self,mapper,size,guard=None,legacy=False):
        self.mapper=mapper;self.size=np.array(size,float);self.guard=guard or PoseGuard();self.filter=OneEuro()
        self.offset=np.zeros(2);self.last=None;self.dot=None;self.raw=None;self.legacy=legacy
    def reset(self):self.filter.reset();self.dot=None;self.last=None
    def advance(self,now):
        if self.last is not None and now-self.last>.25:self.reset()
    def receive(self,o,now):
        self.advance(now)
        features=o.legacy_features if self.legacy else observation_features(self.mapper,o)
        if len(features)==0 or (not self.legacy and not o.valid) or now-o.timestamp>.25 or not self.guard.update(o.pose,now):self.reset();return None
        if self.last is not None and o.timestamp<=self.last:return None
        self.raw=self.mapper.predict(features)*self.size+self.offset
        if self.legacy:
            bounded=np.clip(self.raw,0,self.size)
            filtered=bounded if self.dot is None else self.dot*.6+bounded*.4
        else:filtered=self.filter.apply(self.raw,o.timestamp)
        self.last=o.timestamp
        self.dot=np.clip(filtered,0,self.size) if filtered is not None else None
        return self.dot

def drift_proposal(groups,size,prior_spread):
    if len(groups)!=5 or any(len(g['predictions'])<8 for g in groups):raise ValueError('Incomplete five-point check')
    offsets=np.array([(np.array(g['target'])-np.mean(g['predictions'],axis=0))*size for g in groups])
    center=offsets[0];spread=np.sqrt(np.mean(np.sum((offsets-offsets.mean(0))**2,axis=1)))
    if spread>max(10.,2*prior_spread) or np.linalg.norm(center)>.1*np.linalg.norm(size):raise ValueError('Drift depends on location; recalibrate')
    return center

def confirm_drift(groups,size,offset):
    if len(groups)!=5:return False
    for g in groups:
        p=np.array(g['predictions'])*size;target=np.array(g['target'])*size
        if len(p)<8 or g.get('availability',0)<.8:return False
        spread=np.sqrt(np.mean(np.sum((p-p.mean(0))**2,axis=1)))
        if np.linalg.norm(p+offset-target,axis=1).mean()>np.linalg.norm(p-target,axis=1).mean()+spread:return False
    return True

def geometry_features(o):
    origin=np.array(o.origin);direction=np.array(o.direction)
    if origin.shape!=(3,) or direction.shape!=(3,) or abs(direction[2])<.1:raise ValueError('Invalid ray')
    t=-origin[2]/direction[2]
    if t<=0:raise ValueError('Backward ray')
    xy=(origin+t*direction)[:2]
    return np.r_[xy,origin,o.pose[:3]]


def observation_features(mapper,o):
    if mapper.kind=='geometry':
        try:return geometry_features(o)
        except (ValueError,IndexError):return ()
    return o.features
