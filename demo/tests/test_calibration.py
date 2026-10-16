import sys,unittest,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from calibration import *
from tracking import *
from session_io import *

def obs(t,f=(0.,0.,0.,0.),reason='',seq=None):
    return Observation('r',int(t*1000) if seq is None else seq,t,t,features=tuple(f),reason=reason,pose=(0,0,0,0,0,600),openness=(.2,.2))

class CollectionTests(unittest.TestCase):
    def test_dropout_clears_actual_regression(self):
        s=FixationSampler();s.begin(0)
        for t in np.arange(.6,1.1,.05):s.accept(obs(t),t)
        self.assertTrue(s.samples)
        s.accept(obs(1.1,reason='blink'),1.1)
        self.assertEqual(s.samples,[])
        for t in np.arange(1.5,2.,.05):self.assertFalse(s.accept(obs(t,(1,1,1,1)),t))
        self.assertLess(len(s.samples),20)
    def test_silence_and_queue(self):
        q=ObservationQueue(3)
        for o in [obs(.7),obs(.8,reason='blink'),obs(.9)]:q.put(o)
        items,lost=q.drain();self.assertEqual([o.reason for o in items],['','blink','']);self.assertEqual(lost,0)
        for i in range(5):q.put(obs(i))
        items,lost=q.drain();self.assertEqual(lost,2)
        s=FixationSampler();s.begin(0);s.accept(obs(.7),.7);s.advance(1);self.assertEqual(s.samples,[])
    def test_stale_run_and_coverage(self):
        s=Session('r',(1000,1000));s.presented(0)
        s.receive(Observation('old',1,.8,.8,features=(0,0,0,0)),.8);self.assertEqual(s.sampler.samples,[])
        order=target_order(5);self.assertEqual(len(order),26);self.assertEqual(order[0],4);self.assertEqual(order[-1],4)
        self.assertTrue(all(order.count(i)==2 for i in range(13)))
    def test_stable_rates(self):
        for fps in (15,30,60):
            s=FixationSampler();s.begin(0);ready=False
            for t in np.arange(.6,1.6,1/fps):ready=s.accept(obs(t),t) or ready
            self.assertTrue(ready)

class MappingTests(unittest.TestCase):
    def test_metrics_median_trap(self):
        p=[[.4,.5],[.6,.5]]*10
        m=metrics([dict(target=(.5,.5),predictions=p)],np.array([1000,1000]),1)
        self.assertAlmostEqual(m['mean_error'],100);self.assertAlmostEqual(m['targets'][0]['spread'],100)
        self.assertEqual(m['targets'][0]['bias'],[0.,0.])
        self.assertIsNone(metrics([dict(target=(.5,.5),predictions=[])],np.ones(2),1)['mean_error'])
    def test_weighted_tail(self):
        self.assertEqual(weighted_quantile([1]*10+[100],[.1]*10+[1]),100)
        self.assertEqual(weighted_quantile([1]*20+[100],[.05]*20+[1]),100)
    def test_affine_selection_and_profile(self):
        visits=[]
        for i,target in enumerate(ANCHORS):
            for repeat in range(2):
                f=(target[0]-.5,target[1]-.5,0,0)
                visits.append(Visit(target,[obs(1+j*.05,f) for j in range(20)],location=i))
        m,scores=select_mapper(visits,(1000,1000));self.assertEqual(m.kind,'gaze')
        np.testing.assert_allclose(m.predict((.2,.1,0,0)),(.7,.6),atol=.003)
        np.testing.assert_allclose(Mapper.from_dict(m.to_dict()).predict((.2,.1,0,0)),m.predict((.2,.1,0,0)))
        config=dict(camera_id='c',screen_id='s',framing='off')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'profile.json';save_profile(path,m,config,None,{},[0,0]);load_profile(path,config)
            with self.assertRaises(ValueError):load_profile(path,{**config,'camera_id':'other'})
    def test_geometry_and_pose(self):
        for z,x in [(500,100),(650,130)]:
            o=Observation('r',1,1,1,origin=(0,0,z),direction=(.2,0,-1),pose=(0,0,0,0,0,z))
            self.assertAlmostEqual(geometry_features(o)[0],x)
        guard=PoseGuard([(0,0,0,0,0,600)]);self.assertTrue(guard.update((0,0,0,0,0,800),0))
        self.assertFalse(guard.update((0,0,0,0,0,800),.4));self.assertFalse(guard.update((0,0,0,0,0,600),.5));self.assertTrue(guard.update((0,0,0,0,0,600),.9))
    def test_drift_and_sparse_confirmation(self):
        groups=[dict(target=t,predictions=[np.array(t)-[.02,.01]]*10,availability=1) for t in QUICK]
        offset=drift_proposal(groups,np.array([1000,1000]),1)
        np.testing.assert_allclose(offset,[20,10]);self.assertTrue(confirm_drift(groups,np.array([1000,1000]),offset))
        groups[0]['availability']=.1;self.assertFalse(confirm_drift(groups,np.array([1000,1000]),offset))
    def test_filter_clock(self):
        f=OneEuro();np.testing.assert_allclose(f.apply([10,20],1),[10,20]);self.assertIsNone(f.apply([0,0],1))
        np.testing.assert_allclose(f.apply([100,200],2),[100,200])
        for t in np.arange(2.01,3,.01):np.testing.assert_allclose(f.apply([100,200],t),[100,200])

if __name__=='__main__':unittest.main()

class RobustnessTests(unittest.TestCase):
    def test_no_signal_fit_and_corrupt_json(self):
        with self.assertRaises(ValueError):Mapper().fit(np.zeros((13,4)),ANCHORS)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.json';atomic_json(path,{'good':1})
            with self.assertRaises(ValueError):atomic_json(path,{'bad':float('nan')})
            self.assertEqual(read_json(path),{'good':1})
    def test_filter_matches_rate_and_response(self):
        rng=np.random.default_rng(42)
        for hz in (15,30,60):
            f=OneEuro();raw=rng.normal(0,8,(hz*8,2));adaptive=[];legacy=[];old=None
            for i,value in enumerate(raw):
                adaptive.append(f.apply(value,i/hz));old=value if old is None else .4*value+.6*old;legacy.append(old.copy())
            a=np.array(adaptive)[hz:];b=np.array(legacy)[hz:]
            self.assertLess(np.std(a),.8*np.std(b))
            f=OneEuro();f.apply([0,0],0);old=np.zeros(2);at=None;bt=None
            for i in range(1,hz*2):
                y=f.apply([500,0],i/hz);old=.4*np.array([500,0])+.6*old
                if y[0]>=450 and at is None:at=i/hz
                if old[0]>=450 and bt is None:bt=i/hz
            self.assertLessEqual(at,bt+.1)
    def test_frontoparallel_camera_rejected(self):
        import cv2
        from calibrate_camera import board_points,calibrate_views
        obj=board_points(9,6,25);K=np.array([[1000.,0,640],[0,1000,360],[0,0,1]])
        corners=[]
        for i in range(20):
            pts,_=cv2.projectPoints(obj,np.zeros(3),np.array([-100.+i,-70.,800.+i*5]),K,np.zeros(5));corners.append(pts)
        with self.assertRaises(ValueError):calibrate_views([obj]*20,corners,(1280,720))
