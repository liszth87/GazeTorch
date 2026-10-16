"""Optional checkerboard calibration. Images remain in memory only."""
import argparse
import numpy as np
import cv2
from session_io import atomic_json,camera_identity

def board_points(columns,rows,square):
    obj=np.zeros((columns*rows,3),np.float32);obj[:,:2]=np.mgrid[:columns,:rows].T.reshape(-1,2)*square;return obj

def calibrate_views(objects,corners,size):
    if len(corners)<20:raise ValueError('Need at least 20 distinct board views')
    rms,K,D,rotations,translations=cv2.calibrateCamera(objects,corners,size,None,None)
    tilts=np.array([cv2.Rodrigues(r)[0][:,2] for r in rotations])
    # Front-facing views cannot establish focal length even with perfect residuals.
    if np.ptp(tilts[:,0])<.25 or np.ptp(tilts[:,1])<.25:raise ValueError('Tilt the board about both axes; front-facing views are degenerate')
    centers=np.array([c.reshape(-1,2).mean(0)/size for c in corners])
    if np.any(np.ptp(centers,axis=0)<.35):raise ValueError('Cover more of the image, including its edges')
    scales=np.array([np.linalg.norm(np.ptp(c.reshape(-1,2),axis=0)) for c in corners])
    if scales.max()/scales.min()<1.3:raise ValueError('Vary the board distance/scale')
    subsets=[]
    for parity in (0,1):
        _,sub,_,_,_=cv2.calibrateCamera(objects[parity::2],corners[parity::2],size,None,None);subsets.append(sub)
    relative=np.max(np.abs(np.array([subsets[0][0,0],subsets[0][1,1]])/np.array([subsets[1][0,0],subsets[1][1,1]])-1))
    if not np.isfinite(K).all() or relative>.1:raise ValueError('Intrinsics unstable between view subsets; collect a more diverse set')
    return K,D,dict(training_rms=float(rms),subset_focal_difference=float(relative),tilt_range=np.ptp(tilts,axis=0).tolist())

def validate_profile(d):
    if d.get('schema')!=2 or d.get('type')!='camera-intrinsics' or not d.get('camera_id') or d.get('framing')!='off':raise ValueError('Unsupported or unidentified camera profile')
    for name in ('K','K_new'):
        k=np.asarray(d[name],float)
        if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):raise ValueError('Invalid camera matrix')
    dist=np.array(d['distortion'],float)
    if dist.size not in (4,5,8,12,14) or not np.isfinite(dist).all():raise ValueError('Invalid distortion')
    if len(d['frame_size'])!=2 or min(d['frame_size'])<=0:raise ValueError('Invalid resolution')
    if d.get('diagnostics',{}).get('check_median',float('inf'))>=1.5:raise ValueError('Missing or failed independent board check')
    return d

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--camera',type=int,default=0);parser.add_argument('--columns',type=int,default=9);parser.add_argument('--rows',type=int,default=6);parser.add_argument('--square-mm',type=float,required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    if args.square_mm<=0:parser.error('Square size must be positive')
    ident,framing=camera_identity(args.camera)
    if not ident or framing!='off':raise ValueError('Camera identity/framing unknown or Center Stage active. Disable auto-framing before calibration.')
    obj=board_points(args.columns,args.rows,args.square_mm);objects=[];corners=[];checks=[];K=None
    cap=cv2.VideoCapture(args.camera,cv2.CAP_AVFOUNDATION);cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    try:
        if not cap.isOpened():raise RuntimeError('Camera unavailable')
        while True:
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera read failed')
            if camera_identity(args.camera)!=(ident,framing):raise RuntimeError('Camera/framing changed')
            size=(frame.shape[1],frame.shape[0]);gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            found,pts=cv2.findChessboardCornersSB(gray,(args.columns,args.rows))
            shown=frame.copy()
            if found:cv2.drawChessboardCorners(shown,(args.columns,args.rows),pts,found)
            text=f'Space: capture {len(corners)}/20. Tilt, move, vary distance. Esc: cancel' if K is None else f'Space: fresh check {len(checks)}/5. Esc: cancel'
            cv2.putText(shown,text,(20,30),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),1);cv2.imshow('Camera intrinsics',shown)
            key=cv2.waitKey(20)&255
            if key==27:return
            if key!=32 or not found:continue
            if any(np.sqrt(np.mean((pts-c)**2))<15 for c in corners):print('Move/tilt the board before capturing again');continue
            if K is None:
                corners.append(pts);objects.append(obj.copy())
                if len(corners)>=20:
                    try:K,D,diag=calibrate_views(objects,corners,size)
                    except ValueError as e:print(e);continue
            else:
                train=np.arange(len(obj))%2==0
                ok,r,t=cv2.solvePnP(obj[train],pts[train],K,D)
                if not ok:continue
                projected,_=cv2.projectPoints(obj[~train],r,t,K,D)
                checks.append(float(np.sqrt(np.mean(np.sum((projected-pts[~train])**2,axis=2)))))
                if len(checks)==5:
                    diag['check_median']=float(np.median(checks));diag['check_errors']=checks
                    profile=dict(schema=2,type='camera-intrinsics',camera_id=ident,framing=framing,frame_size=size,K=K.tolist(),K_new=K.tolist(),distortion=D.tolist(),diagnostics=diag,square_mm=args.square_mm)
                    validate_profile(profile);atomic_json(args.output,profile);print(args.output);return
    finally:cap.release();cv2.destroyAllWindows()
if __name__=='__main__':main()
