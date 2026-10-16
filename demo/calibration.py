"""Numerical calibration policy, grouped fitting, and honest measurements.

No Qt or camera dependency. Times are monotonic seconds, targets normalized,
and reported errors are logical screen points.
"""
from dataclasses import dataclass, field
from collections import deque
import threading
import uuid
import numpy as np

ANCHORS = [(x,y) for y in (.07,.5,.93) for x in (.07,.5,.93)] + [(x,y) for y in (.28,.72) for x in (.28,.72)]
VALIDATION = [(.04,.04),(.5,.1),(.96,.04),(.1,.5),(.5,.5),(.9,.5),(.04,.96),(.5,.9),(.96,.96)]
QUICK = [(.5,.5),(.2,.2),(.8,.2),(.2,.8),(.8,.8)]
LEGACY = [(.15,.15),(.5,.15),(.85,.15),(.85,.5),(.5,.5),(.15,.5),(.15,.85),(.5,.85),(.85,.85)]

@dataclass(frozen=True)
class Observation:
    run: str
    sequence: int
    timestamp: float
    completed: float
    features: tuple = ()
    pose: tuple = ()
    origin: tuple = ()
    direction: tuple = ()
    openness: tuple = ()
    reason: str = ''
    frame_size: tuple = ()
    quality: tuple = ()  # mean gray intensity, Laplacian variance
    legacy_features: tuple = ()
    delivered: float | None = None
    @property
    def valid(self):
        return not self.reason and len(self.features)==4 and np.isfinite(self.features).all()

class ObservationQueue:
    """Keep interruptions ordered; overflow is an explicit discontinuity."""
    def __init__(self, capacity=64):
        self.capacity=capacity; self.items=deque(); self.lost=0; self.lock=threading.Lock()
    def put(self, item):
        with self.lock:
            if len(self.items)>=self.capacity:
                self.items.popleft(); self.lost+=1
            self.items.append(item)
    def drain(self):
        with self.lock:
            items=list(self.items); lost=self.lost
            self.items.clear(); self.lost=0
        return items,lost

def target_order(seed=1):
    rng=np.random.default_rng(seed); center=ANCHORS.index((.5,.5))
    others=[i for i in range(13) if i!=center]
    return [center]+list(rng.permutation(others))+list(rng.permutation(others))+[center]

@dataclass
class Visit:
    target: tuple
    observations: list
    role: str='calibration'
    attempt: int=0
    location: int=0
    @property
    def center(self): return np.median([o.features for o in self.observations],axis=0)

class FixationSampler:
    def __init__(self):
        self.samples=[]; self.presented=0.; self.ready_at=0.; self.last=None
        self.eye_reference=None;self.eye_closed=False
        self.feature_limits=np.array([.10,.10,.05,.05]);self.hold_required=False;self.holding=False
    def begin(self, now):
        self.samples=[]; self.presented=now; self.ready_at=now+.6; self.last=None
    def interrupt(self, now):
        self.samples=[]; self.ready_at=max(self.ready_at,now+.35); self.last=None
    def advance(self, now):
        if self.last is not None and now-self.last>.2: self.interrupt(now)
    def accept(self, obs, now):
        if not obs.valid or now-obs.timestamp>.25 or obs.timestamp<self.presented:
            self.interrupt(now); return False
        if self.last is not None and obs.timestamp<=self.last: return False
        if self.last is not None and obs.timestamp-self.last>.2: self.interrupt(obs.timestamp)
        self.last=obs.timestamp
        if obs.timestamp<self.ready_at: return False
        if self.eye_reference is not None and obs.openness:
            threshold=max(.06,self.eye_reference*(.60 if self.eye_closed else .45))
            self.eye_closed=min(obs.openness)<threshold
            if self.eye_closed:self.interrupt(now);return False
        if self.hold_required and not self.holding:return False
        if obs.quality and (obs.quality[0]<20 or obs.quality[0]>240 or obs.quality[1]<3):
            self.interrupt(now); return False
        trial=self.samples+[obs]
        # Robust range bounds prevent slow drift being hidden by a small variance.
        f=np.array([o.features for o in trial])
        if len(trial)>3 and np.any(np.ptp(f,axis=0)>self.feature_limits):
            self.interrupt(now); return False
        if len(trial)>3 and obs.pose and all(o.pose for o in trial):
            poses=np.array([o.pose for o in trial])
            if np.any(np.ptp(poses,axis=0)>[.12,.12,.12,45,45,65]):
                self.interrupt(now); return False
        self.samples.append(obs)
        return len(self.samples)>=8 and obs.timestamp-self.samples[0].timestamp>=.8

def basis(x, kind):
    x=np.asarray(x,dtype=float)
    if kind=='gaze': return x[...,:2]
    if kind=='affine': return x[...,:4]
    if kind=='quadratic':
        a,b=x[...,0],x[...,1]
        return np.stack([a,b,a*a,a*b,b*b],axis=-1)
    if kind=='geometry': return x
    raise ValueError('Unknown mapping')

class Mapper:
    def __init__(self, kind='affine', penalty=.01): self.kind=kind; self.penalty=float(penalty)
    def fit(self,x,y,weights=None,robust=True,legacy=False):
        x=basis(x,self.kind);y=np.asarray(y,float)
        if x.ndim!=2 or len(x)<3 or y.shape!=(len(x),2) or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Invalid calibration observations')
        if np.max(np.ptp(x,axis=0))<1e-7:raise ValueError('No usable gaze variation across calibration')
        self.mean=x.mean(0);self.scale=np.maximum(x.std(0),.005)
        a=np.c_[np.ones(len(x)),(x-self.mean)/self.scale]
        w=np.ones(len(x)) if weights is None else np.asarray(weights,float)
        base=w/w.sum(); w=base.copy()
        reg=np.eye(a.shape[1])*self.penalty;reg[0,0]=0
        if legacy: w=np.ones(len(x))
        for _ in range(3 if robust and not legacy else 1):
            self.coef=np.linalg.solve(a.T@(w[:,None]*a)+reg,a.T@(w[:,None]*y))
            residual=np.linalg.norm(a@self.coef-y,axis=1)
            cutoff=max(float(np.median(residual))*1.5,.005)
            w=base*np.clip(cutoff/np.maximum(residual,1e-9),.25,1)
            # Preserve target totals (two consecutive visits per target in fitter).
            if weights is not None and len(w)%2==0:
                for i in range(0,len(w),2): w[i:i+2]*=base[i:i+2].sum()/w[i:i+2].sum()
        if not np.isfinite(self.coef).all(): raise ValueError('Nonfinite fit')
        return self
    def predict(self,x):
        z=(basis(x,self.kind)-self.mean)/self.scale
        return np.concatenate([np.ones(z.shape[:-1]+(1,)),z],axis=-1)@self.coef
    def to_dict(self):
        return dict(kind=self.kind,penalty=self.penalty,mean=self.mean.tolist(),scale=self.scale.tolist(),coef=self.coef.tolist())
    @classmethod
    def from_dict(cls,d):
        m=cls(d['kind'],d['penalty']);m.mean=np.array(d['mean'],float);m.scale=np.array(d['scale'],float);m.coef=np.array(d['coef'],float)
        n={'gaze':2,'affine':4,'quadratic':5,'geometry':8}.get(m.kind)
        if n is None or m.mean.shape!=(n,) or m.scale.shape!=(n,) or m.coef.shape!=(n+1,2) or np.any(m.scale<=0) or not all(np.isfinite(v).all() for v in (m.mean,m.scale,m.coef)):
            raise ValueError('Invalid mapping profile')
        return m

def weighted_quantile(values,weights,q=.9):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    if not len(values) or weights.sum()<=0: return None
    order=np.argsort(values); cumulative=np.cumsum(weights[order])/weights.sum()
    return float(values[order[min(np.searchsorted(cumulative,q),len(order)-1)]])

def metrics(groups,size,required=None):
    """Each group is a required location with target and raw/display predictions."""
    rows=[]; all_errors=[];weights=[]
    for g in groups:
        p=np.asarray(g.get('predictions',[]),float).reshape(-1,2)
        if not np.isfinite(p).all(): raise ValueError('Nonfinite predictions')
        target=np.array(g['target']); diff=(p-target)*size
        err=np.linalg.norm(diff,axis=1)
        mean=diff.mean(0) if len(p) else None
        spread=float(np.sqrt(np.mean(np.sum((diff-mean)**2,axis=1)))) if len(p) else None
        rows.append(dict(target=list(target),count=len(p),mean_error=float(err.mean()) if len(p) else None,
                         p90=float(np.quantile(err,.9)) if len(p) else None,bias=mean.tolist() if len(p) else None,spread=spread))
        if len(err): all_errors.extend(err);weights.extend([1/len(err)]*len(err))
    complete=bool(rows) and all(r['count'] for r in rows) and (required is None or len(rows)==required)
    perimeter=[r['mean_error'] for r in rows if min(r['target'])<=.1 or max(r['target'])>=.9]
    return dict(complete=complete,targets=rows,mean_error=float(np.mean([r['mean_error'] for r in rows])) if complete else None,
                p90=weighted_quantile(all_errors,weights) if complete else None,
                perimeter_mean=float(np.mean(perimeter)) if complete and perimeter else None)

def select_mapper(visits,size):
    groups={}
    for v in visits: groups.setdefault(v.target,[]).append(v)
    if len(groups)!=13 or any(len(v)!=2 for v in groups.values()): raise ValueError('Need two visits at all 13 locations')
    scores=[]; best_by_kind=[]
    def fit_for(train,kind,penalty):
        ordered=[v for key in sorted(train) for v in train[key]]
        return Mapper(kind,penalty).fit([v.center for v in ordered],[v.target for v in ordered],np.ones(len(ordered)))
    for kind in ('gaze','affine','quadratic'):
        candidates=[]
        for penalty in (.001,.01,.1,1.):
            held=[]
            for target,vs in groups.items():
                m=fit_for({k:v for k,v in groups.items() if k!=target},kind,penalty)
                obs=[o.features for v in vs for o in v.observations]
                held.append(dict(target=target,predictions=m.predict(obs).tolist()))
            score=metrics(held,np.array(size),13);score.update(kind=kind,penalty=penalty);scores.append(score);candidates.append(score)
        best_by_kind.append(min(candidates,key=lambda s:s['mean_error']))
    winner=best_by_kind[0]
    for candidate in best_by_kind[1:]:
        if (candidate['mean_error']<=winner['mean_error']*.9 and candidate['p90']<=winner['p90']*1.05 and candidate['perimeter_mean']<=winner['perimeter_mean']*1.05):winner=candidate
    return fit_for(groups,winner['kind'],winner['penalty']),scores

class Session:
    """Own target progression and contiguous collection; view only renders state."""
    def __init__(self,run,size,seed=1,legacy=False):
        self.id=uuid.uuid4().hex;self.run=run;self.size=np.array(size);self.seed=seed;self.legacy=legacy
        self.order=list(range(9)) if legacy else target_order(seed)
        self.targets=LEGACY if legacy else ANCHORS
        self.index=0;self.attempt=0;self.visits=[];self.attempts=[];self.sampler=FixationSampler()
        self.state='preflight';self.started=0.;self.last_sequence=-1;self.dropped=0
        self.preflight=[];self.retry_state='preflight';self.mapper=None;self.scores=[];self.validation=[];self.samples=[];self.failure=''
        self.role='calibration';self.report=None;self.present_pending=True;self.completion_state='fitting';self.cues=[]
    @property
    def target(self):
        if self.state=='preflight':return (.5,.5)
        return self.targets[self.order[self.index]] if self.index<len(self.order) else (.5,.5)
    def presented(self,now):
        if self.present_pending:
            self.started=now;self.sampler.begin(now);self.present_pending=False
    def begin_validation(self,mapper,targets=VALIDATION,role='validation'):
        self.mapper=mapper;self.targets=targets;self.order=list(range(len(targets)));self.index=0;self.attempt=0
        self.state='validating';self.role=role;self.samples=[];self.validation=[];self.present_pending=True
    def overflow(self,count,now):
        self.dropped+=count;self.sampler.interrupt(now)
    def advance(self,now):
        if self.present_pending:return
        self.sampler.advance(now)
        if self.state in ('preflight','collecting') and now-self.started>=6:
            self.retry(now,'Could not obtain a stable fixation')
        if self.state=='validating' and now-self.started>=2.1:
            scored=[o for o in self.samples if self.started+.6<=o.timestamp<self.started+2.1]
            valid=[o for o in scored if o.valid]
            gaps=np.diff([self.started+.6]+[o.timestamp for o in valid]+[self.started+2.1])
            interrupted=len(valid)<8 or (len(gaps)>0 and max(gaps)>.35)
            self.validation.append(dict(target=self.target,observations=self.samples.copy(),started=self.started,ended=self.started+2.1,failed_attempt=interrupted))
            self.samples=[]
            if interrupted:
                self.retry(now,'Accuracy check interrupted. Retry this target.');return
            self.index+=1;self.attempt=0;self.present_pending=True
            if self.index>=len(self.order):self.state='results'
    def retry(self,now,reason):
        self.retry_state=self.state
        self.attempts.append(dict(target=self.target,attempt=self.attempt,reason=reason));self.attempt+=1
        self.sampler.begin(now);self.started=now;self.failure=reason
        if self.attempt>2:self.state='retry-needed'
        else:self.present_pending=True
    def manual_retry(self):
        self.attempt=0;self.state=self.retry_state;self.present_pending=True
    def receive(self,o,now):
        if o.run!=self.run or o.sequence<=self.last_sequence:return
        self.last_sequence=o.sequence
        if self.present_pending:return
        if self.state=='validating':
            if self.started<=o.timestamp<self.started+2.1:self.samples.append(o)
            return
        if self.state not in ('preflight','collecting'):return
        if self.legacy and self.state=='collecting':
            if o.legacy_features and now-self.started>.9:self.sampler.samples.append(o)
            ready=len(self.sampler.samples)>=20
        else:ready=self.sampler.accept(o,now)
        if not ready:return
        if self.state=='preflight':
            self.preflight=self.sampler.samples.copy()
            eyes=[min(x.openness) for x in self.preflight if x.openness]
            self.sampler.eye_reference=float(np.median(eyes)) if eyes else None
            features=np.array([x.features for x in self.preflight]);noise=np.median(np.abs(features-np.median(features,axis=0)),axis=0)*1.4826
            self.sampler.feature_limits=np.clip(8*noise,[.03,.03,.015,.015],[.10,.10,.05,.05])
            self.state='collecting';self.present_pending=True;self.attempt=0;return
        prior=[v for v in self.visits if v.target==tuple(self.target)]
        if prior and not self.legacy and self.role=='calibration':
            center=np.median([o.features for o in self.sampler.samples],axis=0)
            if np.any(np.abs(center-prior[0].center)>[.08,.08,.05,.05]):
                self.retry(now,'Repeated visit disagrees. Look at the center and retry.');return
        self.visits.append(Visit(tuple(self.target),self.sampler.samples.copy(),attempt=self.attempt,location=self.order[self.index],role=self.role))
        self.index+=1;self.attempt=0;self.present_pending=True
        if self.index>=len(self.order):self.state=self.completion_state
