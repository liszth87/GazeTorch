"""Replay exported numeric sessions or run explicit synthetic checks."""
import argparse
from dataclasses import asdict
import numpy as np
from calibration import Observation, metrics, Mapper
from tracking import Tracker, PoseGuard, observation_features
from session_io import atomic_json, read_json

def score_validation(visits,mapper,size,guard=None,legacy=False,offset=(0,0)):
    tracker=Tracker(mapper,size,guard,legacy);tracker.offset=np.array(offset,float)
    raw=[];display=[];availability=[];invalid=0;observed=0;exports=[];settling=[]
    for visit in visits:
        start=visit['started']+.6;end=visit['ended'];r=[];d=[];display_times=[];available=0.;previous=start
        # Reset only at start of whole sequence, not every target.
        for o in visit['observations']:
            t=o.delivered if o.delivered is not None else o.completed
            if t>=start:
                until=min(t,end)
                if tracker.dot is not None and tracker.last is not None:
                    available+=max(0,min(until,tracker.last+.25)-max(previous,start))
                previous=max(previous,until)
            tracker.receive(o,t)
            if start<=o.timestamp<end:
                observed+=1
                features=o.legacy_features if legacy else observation_features(mapper,o)
                if len(features) and (legacy or o.valid):r.append((mapper.predict(features)+np.array(offset)/size).tolist())
                else:invalid+=1
                if tracker.dot is not None:
                    d.append((tracker.dot/size).tolist());display_times.append(t)
        if tracker.dot is not None and tracker.last is not None:available+=max(0,min(end,tracker.last+.25)-max(previous,start))
        tracker.advance(end)
        fraction=min(1.,available/(end-start));availability.append(fraction)
        target_delta=None
        if len(d)>1:
            points=np.array(d)*size;final=np.mean(points[-max(1,len(points)//3):],axis=0)
            radius=max(1.,np.linalg.norm(points[0]-final)*.1)
            hits=np.flatnonzero(np.linalg.norm(points-final,axis=1)<=radius)
            if len(hits):target_delta=float(display_times[hits[0]]-visit['started'])
        settling.append(target_delta)
        raw.append(dict(target=visit['target'],predictions=r,availability=fraction))
        display.append(dict(target=visit['target'],predictions=d))
        exports.append(dict(target=visit['target'],started=visit['started'],ended=end,failed_attempt=visit.get('failed_attempt',False),observations=[asdict(o) for o in visit['observations']]))
    # Reattempts remain in the target's distribution, never become extra locations.
    def merge(rows):
        merged={}
        for row in rows:
            key=tuple(row['target']);item=merged.setdefault(key,dict(target=key,predictions=[],availability=[]))
            item['predictions'].extend(row['predictions']);item['availability'].append(row.get('availability',0))
        for item in merged.values():item['availability']=float(np.mean(item['availability']))
        return list(merged.values())
    raw=merge(raw);display=merge(display)
    report=dict(schema=2,type='validation',synthetic=False,raw=metrics(raw,np.array(size),len(raw)),displayed=metrics(display,np.array(size),len(raw)),
                time_availability=[g['availability'] for g in raw],attempt_time_availability=availability,target_to_dot_settling_estimate=settling,observed_frames=observed,invalid_frames=invalid,backend_drops=None)
    return report,raw,exports

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--synthetic',action='store_true');g.add_argument('--session');p.add_argument('--output',required=True);a=p.parse_args()
    if a.synthetic:
        groups=[dict(target=(.5,.5),predictions=[[.4,.5],[.6,.5]]*20)]
        result=dict(schema=2,synthetic=True,report=metrics(groups,np.array([1000,1000]),1))
    else:
        data=read_json(a.session)
        if data.get('schema')!=2 or data.get('synthetic') is not False:raise ValueError('Expected live schema-2 export')
        blocks=data.get('blocks',[data]);results=[]
        for block in blocks:
            mapper=Mapper.from_dict(block['mapper']);visits=[]
            for v in block['validation']:
                visits.append({**v,'observations':[Observation(**o) for o in v['observations']]})
            report,_,_=score_validation(visits,mapper,np.array(block['size']),guard=PoseGuard(limits=block.get('pose')),legacy=block.get('legacy',False),offset=block.get('offset',[0,0]))
            if not report['raw']['complete']:raise ValueError('Incomplete validation')
            results.append(report)
        result=dict(schema=2,synthetic=False,blocks=results)
    atomic_json(a.output,result)
if __name__=='__main__':main()
