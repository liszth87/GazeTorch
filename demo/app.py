"""Native Qt view for the numerical calibration and tracking policies."""
import sys,time,uuid
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import cv2
from PySide6.QtCore import Qt,QThread,Signal,QTimer,QPointF,QRectF
from PySide6.QtGui import QPainter,QColor,QFont,QImage,QPixmap,QShortcut,QKeySequence,QTransform
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QSpinBox,QComboBox,QFileDialog,QMessageBox
from engine import GazeEngine,ROOT
from calibration import Observation,ObservationQueue,Session,Mapper,select_mapper,QUICK
from tracking import Tracker,PoseGuard,drift_proposal,confirm_drift,geometry_features,observation_features
from session_io import DATA,atomic_json,save_profile,load_profile,camera_identity,fingerprint,read_json
from evaluate_calibration import score_validation

class CameraWorker(QThread):
    sample=Signal(object,object,float)  # compatibility with real-worker regression
    status=Signal(str)
    def __init__(self,index,camera_profile=None):
        super().__init__();self.index=index;self.camera_profile=camera_profile;self.run_id=uuid.uuid4().hex
        self.queue=ObservationQueue();self.preview=None;self.identity=None;self.framing='unknown';self.frame_size=None
        self.setStackSize(16*1024*1024)
    def run(self):
        cap=None;engine=None;seq=0
        try:
            self.status.emit('Loading GazeTorch…');engine=GazeEngine(self.camera_profile)
            before=camera_identity(self.index)
            cap=cv2.VideoCapture(self.index,cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            if not cap.isOpened():raise RuntimeError('Camera unavailable. Check macOS camera permission or select another camera.')
            after=camera_identity(self.index)
            self.identity,self.framing=after if before==after else (None,'unknown')
            if self.camera_profile and (self.identity!=self.camera_profile['camera_id'] or self.framing!='off'):raise ValueError('Camera profile does not match this camera/framing state')
            failures=0;last_check=time.monotonic()
            while not self.isInterruptionRequested():
                ok,frame=cap.read();stamp=time.monotonic();seq+=1
                if not ok:
                    self.queue.put(Observation(self.run_id,seq,stamp,stamp,reason='capture failure'))
                    failures+=1
                    if failures>30:raise RuntimeError('Camera stopped delivering frames. Stop and restart.')
                    self.msleep(50);continue
                failures=0;h,w=frame.shape[:2];self.frame_size=(w,h)
                if stamp-last_check>1:
                    current=camera_identity(self.index);last_check=stamp
                    if current!=(self.identity,self.framing) and self.identity is not None:raise RuntimeError('Camera or framing changed. Restart and recalibrate.')
                value=engine.infer(frame);details=engine.details.copy()
                features=tuple(value[0]) if value is not None else ()
                o=Observation(self.run_id,seq,stamp,time.monotonic(),features=features,frame_size=(w,h),**details)
                self.queue.put(o)
                self.preview=cv2.cvtColor(cv2.resize(frame,(320,180)),cv2.COLOR_BGR2RGB)
                self.sample.emit(None if not o.valid else np.array(features),self.preview,(o.completed-stamp)*1000)
        except Exception as e:self.status.emit(str(e))
        finally:
            if cap is not None:cap.release()
            if engine is not None:engine.close()

class FitWorker(QThread):
    result=Signal(str,object,object,str)
    def __init__(self,session):
        super().__init__();self.session=session;self.setStackSize(16*1024*1024)
    def run(self):
        try:
            s=self.session
            if getattr(s,'geometry',False):
                visits=s.base_visits+s.visits
                x=[np.median([geometry_features(o) for o in v.observations],axis=0) for v in visits]
                targets=[v.target for v in visits];counts={t:targets.count(t) for t in targets}
                m=Mapper('geometry',.01).fit(x,targets,[1/counts[t] for t in targets],robust=False);scores=[]
            elif s.legacy:
                x=[np.median([o.legacy_features for o in v.observations],axis=0) for v in s.visits]
                m=Mapper('affine',.3).fit(x,[v.target for v in s.visits],robust=False,legacy=True);scores=[]
            else:m,scores=select_mapper(s.visits,s.size)
            self.result.emit(s.id,m,scores,'')
        except Exception as e:self.result.emit(self.session.id,None,None,str(e))

class Stage(QWidget):
    stopped=Signal();changed=Signal();completed=Signal(object);continue_requested=Signal()
    def __init__(self):
        super().__init__();self.setWindowFlags(Qt.WindowType.Window|Qt.WindowType.FramelessWindowHint|Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.session=None;self.tracker=None;self.desktop=False;self.report=None;self.export=None;self.fitters=[]
        self.quick_phase=None;self.quick_offset=None;self.previous_tracker=None;self.last_observation=None;self.geometry_phase=None;self.geometry_base=None;self.geometry_candidate=None;self.manual_capture=False;self.last_validated=None;self.calibration_visits=[]
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(30)
        self.shortcuts=[]
        for key,fn in [('Escape',self.stop),('D',self.toggle_desktop),('C',self.recalibrate),('R',self.retry),('Return',self.continue_requested.emit),('Space',self.continue_requested.emit)]:
            shortcut=QShortcut(QKeySequence(key),self);shortcut.activated.connect(fn);self.shortcuts.append(shortcut)
    def start(self,run,screen,legacy=False):
        self.setGeometry(screen.geometry());self.show();self.activateWindow()
        self.session=Session(run,(self.width(),self.height()),seed=int(time.time())%100000,legacy=legacy)
        self.tracker=None;self.quick_phase=None;self.report=None;self.desktop=False;self.changed.emit();self.update()
    def recalibrate(self):
        if self.session:self.start(self.session.run,self.screen(),self.session.legacy)
    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_H and self.session:self.session.sampler.holding=True
        else:super().keyPressEvent(event)
    def keyReleaseEvent(self,event):
        if event.key()==Qt.Key.Key_H and self.session:
            self.session.sampler.holding=False
            if self.manual_capture:self.session.sampler.interrupt(time.monotonic())
        else:super().keyReleaseEvent(event)
    def retry(self):
        if self.session and self.session.state=='retry-needed':self.session.manual_retry()
    def cancel(self):
        self.session=None;self.tracker=None;self.quick_phase=None;self.hide();self.changed.emit()
    def stop(self):self.cancel();self.stopped.emit()
    def toggle_desktop(self):
        if self.session and self.session.state=='tracking':self.desktop=not self.desktop;self.update()
    def receive(self,o,now):
        self.last_observation=o
        if not self.session or o.run!=self.session.run:return
        self.session.sampler.hold_required=self.manual_capture and self.session.state=='collecting'
        self.session.receive(o,now)
        if self.session.state=='tracking' and self.tracker:self.tracker.receive(o,now)
    def tick(self):
        if not self.session:return
        now=time.monotonic();s=self.session;s.advance(now)
        if self.tracker:self.tracker.advance(now)
        if s.state=='fitting':
            s.state='fit-pending';f=FitWorker(s);f.result.connect(self.fit_result);f.finished.connect(lambda f=f:self.fitters.remove(f));self.fitters.append(f);f.start()
        if s.state=='quick-proposal-ready':self.finish_quick_proposal()
        if s.state=='results':self.finish_validation()
        self.update()
    def fit_result(self,session_id,mapper,scores,error):
        if not self.session or self.session.id!=session_id:return
        if error:
            self.session.failure=error;self.session.state='retry-needed';return
        self.session.scores=scores
        if not self.geometry_phase:self.calibration_visits=self.session.visits.copy()
        if self.geometry_phase=='collect':
            self.geometry_candidate=mapper;self.geometry_phase='confirm'
            targets=[(.5,.5),(.25,.25),(.75,.25),(.25,.75),(.75,.75)]*4
            from calibration import VALIDATION
            self.session.begin_validation(mapper,targets+VALIDATION,'development')
            self.session.cues=[cue for cue in ('Hold slightly left','Hold slightly right','Hold slightly nearer','Hold slightly farther') for _ in range(5)]+['Return to neutral posture']*9
            return
        poses=[o.pose for v in self.session.visits for o in v.observations if o.pose]
        self.tracker=Tracker(mapper,(self.width(),self.height()),PoseGuard(poses),self.session.legacy)
        self.session.begin_validation(mapper)
    def finish_validation(self):
        s=self.session
        if self.geometry_phase=='confirm':
            candidate,_,_=score_validation(s.validation,self.geometry_candidate,s.size)
            base,_,_=score_validation(s.validation,self.geometry_base.mapper,s.size)
            cpose,_,_=score_validation(s.validation[:20],self.geometry_candidate,s.size)
            bpose,_,_=score_validation(s.validation[:20],self.geometry_base.mapper,s.size)
            cstill,_,_=score_validation(s.validation[20:],self.geometry_candidate,s.size)
            bstill,_,_=score_validation(s.validation[20:],self.geometry_base.mapper,s.size)
            promote=(candidate['raw']['complete'] and base['raw']['complete'] and cpose['raw']['mean_error']<=.9*bpose['raw']['mean_error'] and cstill['raw']['mean_error']<=1.05*bstill['raw']['mean_error'] and cstill['raw']['perimeter_mean']<=1.05*bstill['raw']['perimeter_mean'] and np.mean(candidate['time_availability'])>=np.mean(base['time_availability'])-.02)
            s.scores={'geometry':candidate,'baseline':base,'promoted':bool(promote)}
            poses=[o.pose for v in s.base_visits+s.visits for o in v.observations if o.pose]
            self.tracker=Tracker(self.geometry_candidate,s.size,PoseGuard(poses)) if promote else self.geometry_base
            self.geometry_phase=None;s.cues=[];s.begin_validation(self.tracker.mapper);return
        report,groups,exported=score_validation(s.validation,s.mapper,s.size,guard=PoseGuard(limits=self.tracker.guard.to_list()) if self.tracker else None,legacy=s.legacy,offset=self.tracker.offset if self.tracker else (0,0))
        report.update(session=s.id,seed=s.seed,application_drops=s.dropped,retries=s.attempts)
        if self.quick_phase=='confirmation':
            if not confirm_drift(groups,s.size,self.quick_offset):
                s.failure='Quick check failed. Please recalibrate.';s.state='check-failed';self.tracker.reset();return
            self.tracker.offset+=self.quick_offset;self.quick_phase=None
            report,groups,exported=score_validation(s.validation,s.mapper,s.size,guard=PoseGuard(limits=self.tracker.guard.to_list()),offset=self.tracker.offset)
        self.report=report
        self.last_validated=(self.tracker,self.report) if report['raw']['complete'] else self.last_validated
        self.export=dict(schema=2,synthetic=False,mapper=s.mapper.to_dict(),legacy=s.legacy,size=s.size.tolist(),offset=self.tracker.offset.tolist(),pose=self.tracker.guard.to_list(),validation=exported,
                         selection_scores=s.scores,protocol_version=2,calibration=[dict(target=v.target,role=v.role,observations=[asdict(o) for o in v.observations]) for v in s.visits],report=report)
        s.state='results-view';self.tracker.reset();self.changed.emit();self.completed.emit(self.export)
    def track(self):
        if self.session and self.session.state=='results-view' and self.report['raw']['complete']:
            self.session.state='tracking';self.tracker.reset();self.changed.emit()
    def quick_check(self,run=None,mapper=None,profile=None):
        if mapper is not None:
            self.session=Session(run,(self.width(),self.height()));self.tracker=Tracker(mapper,self.session.size,PoseGuard(limits=profile.get('pose')))
            self.tracker.offset=np.array(profile.get('offset',[0,0]));self.report=profile.get('report')
        if not self.tracker:return
        self.quick_phase='proposal'
        old=self.session
        self.session=Session(old.run,old.size);self.session.mapper=self.tracker.mapper
        self.session.targets=QUICK;self.session.order=list(range(5));self.session.state='collecting';self.session.role='quick-check';self.session.completion_state='quick-proposal-ready'
        self.desktop=False;self.show()
    def finish_quick_proposal(self):
        s=self.session
        groups=[dict(target=v.target,predictions=[s.mapper.predict(observation_features(s.mapper,o))+self.tracker.offset/s.size for o in v.observations]) for v in s.visits]
        try:
            spreads=[r['spread'] for r in (self.report or {}).get('raw',{}).get('targets',[]) if r['spread'] is not None]
            self.quick_offset=drift_proposal(groups,s.size,float(np.median(spreads)) if spreads else 0)
            self.quick_phase='confirmation';s.begin_validation(s.mapper,QUICK,'quick-check')
        except ValueError as e:s.failure=str(e);s.state='check-failed';self.tracker.reset()
    def posture_pack(self):
        if not self.tracker or not self.session or len(self.calibration_visits)!=26:return
        self.geometry_base=self.tracker;self.geometry_phase='collect'
        old=self.session;s=Session(old.run,old.size);s.base_visits=self.calibration_visits;s.geometry=True
        s.targets=[(.5,.5),(.25,.25),(.75,.25),(.25,.75),(.75,.75)]*4;s.order=list(range(20));s.state='collecting';s.role='development'
        s.cues=[cue for cue in ('Hold slightly left','Hold slightly right','Hold slightly nearer','Hold slightly farther') for _ in range(5)]
        self.session=s;self.desktop=False;self.show()
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self.desktop:p.fillRect(self.rect(),QColor('#111722'))
        if not self.session:return
        s=self.session
        if s.present_pending and s.state in ('preflight','collecting','validating'):s.presented(time.monotonic())
        p.setFont(QFont('Helvetica',17));p.setPen(QColor('#edf3fa'))
        if s.state in ('preflight','collecting','validating'):
            title='Get comfortable · look at the center' if s.state=='preflight' else ('Quick check' if self.quick_phase else 'Accuracy check' if s.state=='validating' else 'Calibration')+f'  {s.index+1}/{len(s.order)}'
            if self.manual_capture and s.state=='collecting':title+=' · Hold H when ready'
            if s.cues and s.index<len(s.cues):title+=' · '+s.cues[s.index]
            p.drawText(QRectF(50,65,self.width()-100,35),Qt.AlignmentFlag.AlignCenter,title)
            p.setFont(QFont('Helvetica',12));p.drawText(QRectF(50,105,self.width()-100,35),Qt.AlignmentFlag.AlignCenter,'Look at the small center. Keep your head comfortably still. Esc stops the camera.')
            x,y=np.array(s.target)*[self.width(),self.height()]
            p.setPen(QColor('#7de5c0'));p.setBrush(QColor('#213d3b'));p.drawEllipse(QPointF(x,y),18,18)
            p.setBrush(QColor('#f6fffc'));p.drawEllipse(QPointF(x,y),4,4)
        elif s.state=='tracking':
            if self.tracker and self.tracker.dot is not None:
                x,y=self.tracker.dot;p.setPen(Qt.PenStyle.NoPen);p.setBrush(QColor('#64efc3'));p.drawEllipse(QPointF(x,y),8,8)
            p.setPen(QColor('#edf3fa'));p.drawText(30,40,'D: desktop overlay   C: recalibrate   Esc: stop camera')
            if self.tracker and self.tracker.guard.blocked:p.drawText(30,65,'Return to your calibrated posture.')
        else:
            text={'fit-pending':'Fitting calibration…','retry-needed':s.failure+' · R: retry or Esc: stop','check-failed':s.failure+' · C: recalibrate','results-view':self.result_text()}.get(s.state,s.state)
            p.drawText(QRectF(60,100,self.width()-120,200),Qt.AlignmentFlag.AlignCenter|Qt.TextFlag.TextWordWrap,text)
            if s.state=='results-view' and self.report:
                for row in self.report['raw']['targets']:
                    x,y=np.array(row['target'])*[self.width(),self.height()];value=row['mean_error']
                    p.setPen(QColor('#7de5c0'));p.drawText(QRectF(x-50,y-15,100,30),Qt.AlignmentFlag.AlignCenter,'—' if value is None else f'{value:.0f} pt')
    def result_text(self):
        if not self.report:return 'No result'
        m=self.report['raw'];error=m['mean_error'];avail=np.mean(self.report['time_availability']) if self.report['time_availability'] else 0
        return ('Check incomplete. Recalibrate.' if error is None else f'Mean error: {error:.0f} screen points · 90th percentile: {m["p90"]:.0f}\nUsable time: {avail:.0%}\nPress Enter to show the dot / continue comparison. C recalibrates.')

class Control(QWidget):
    permission=Signal(bool)
    def __init__(self):
        super().__init__();self.setWindowTitle('GazeTorch · Webcam dot');self.setMinimumWidth(550)
        self.permission.connect(self.permission_result);self.worker=None;self.stage=Stage();self.stage.stopped.connect(self.stop);self.stage.changed.connect(self.update_buttons);self.stage.completed.connect(self.results);self.stage.continue_requested.connect(self.track_or_next)
        self.camera_profile=None;self.comparison=None;self.blocks=[];self.config=None
        self.setStyleSheet('QWidget {background:#111722;color:#edf3fa;font-size:14px;} QPushButton {background:#245c50;padding:9px;border-radius:7px;} QLabel {padding:5px;}')
        layout=QVBoxLayout(self);layout.addWidget(QLabel('GazeTorch webcam dot'))
        help=QLabel('Two calibration passes, then an independent accuracy check.\nUse even lighting and turn off camera auto-framing. No video is saved.');layout.addWidget(help)
        self.screen_choice=QComboBox();self.screen_choice.addItems([s.name() for s in QApplication.screens()]);layout.addWidget(self.screen_choice)
        self.camera=QSpinBox();self.camera.setRange(0,5);layout.addWidget(QLabel('Camera index'));layout.addWidget(self.camera)
        self.preview=QLabel();layout.addWidget(self.preview);self.status=QLabel('Camera is off.');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.start_btn=QPushButton('Start camera');self.start_btn.clicked.connect(self.start);layout.addWidget(self.start_btn)
        self.cal_btn=QPushButton('Calibrate');self.cal_btn.clicked.connect(self.begin);layout.addWidget(self.cal_btn)
        self.track_btn=QPushButton('Show dot / next comparison block');self.track_btn.clicked.connect(self.track_or_next);layout.addWidget(self.track_btn)
        stop=QPushButton('Stop camera');stop.clicked.connect(self.stop);layout.addWidget(stop)
        self.advanced=QWidget();advanced_layout=QVBoxLayout(self.advanced)
        for name,fn in [('Hold-to-capture on/off',self.toggle_hold),('Quick check',self.quick),('Posture calibration experiment',self.posture),('Retry target',self.stage.retry),('Cancel calibration',self.cancel),('Save calibration',self.save),('Load calibration',self.load),('Export diagnostics',self.export),('Run comparison',self.compare),('Import camera intrinsics',self.import_camera)]:
            b=QPushButton(name);b.clicked.connect(fn);advanced_layout.addWidget(b)
        toggle=QPushButton('Advanced');toggle.setCheckable(True);toggle.toggled.connect(self.advanced.setVisible);layout.addWidget(toggle);layout.addWidget(self.advanced);self.advanced.hide()
        self.timer=QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(30)
        QApplication.instance().screenRemoved.connect(lambda _:self.setup_changed())
        for screen in QApplication.screens():screen.geometryChanged.connect(lambda _:self.setup_changed())
        self.screen_choice.currentIndexChanged.connect(self.setup_changed);self.camera.valueChanged.connect(self.setup_changed)
        self.update_buttons()
    def screen_selected(self):
        screens=QApplication.screens();return screens[min(self.screen_choice.currentIndex(),len(screens)-1)]
    def configuration(self):
        s=self.screen_selected();g=s.geometry();serial=s.serialNumber()
        return dict(camera_id=self.worker.identity if self.worker else None,framing=self.worker.framing if self.worker else 'unknown',frame_size=list(self.worker.frame_size or ()) if self.worker else [],
                    screen_id='|'.join([s.manufacturer(),s.model(),serial]) if serial else None,screen=[g.x(),g.y(),g.width(),g.height(),s.devicePixelRatio()],
                    intrinsics=self.camera_profile,feature_version=2,source={name:fingerprint(ROOT/'demo'/name) for name in ('engine.py','calibration.py','tracking.py','app.py')},model=fingerprint(ROOT/'demo/assets/GazeTorch_MobileNet.pkl'))
    def setup_changed(self,*args):
        if self.worker:self.stop();self.status.setText('Setup changed. Start camera and recalibrate.')
    def permission_result(self,allowed):
        if allowed:self.start()
        else:self.status.setText('Camera access denied. Enable GazeTorch Demo in System Settings → Privacy & Security → Camera.')
    def start(self):
        if self.worker:return
        import AVFoundation as AV
        status=AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
        if status==0:
            AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo,lambda granted:self.permission.emit(bool(granted)));return
        if status!=3:self.permission_result(False);return
        self.worker=CameraWorker(self.camera.value(),self.camera_profile);self.worker.status.connect(self.status.setText);self.worker.finished.connect(self.finished);self.worker.start();self.update_buttons()
    def poll(self):
        if not self.worker:return
        observations,lost=self.worker.queue.drain();now=time.monotonic()
        if self.stage.session and lost:self.stage.session.overflow(lost,now)
        for o in observations:self.stage.receive(replace(o,delivered=now),now)
        preview=self.worker.preview
        if preview is not None:
            h,w,_=preview.shape;image=QImage(preview.data,w,h,3*w,QImage.Format.Format_RGB888).copy()
            self.preview.setPixmap(QPixmap.fromImage(image).transformed(QTransform().scale(-1,1)))
        if observations and not self.stage.session:self.status.setText('Eyes visible. Ready to calibrate.' if observations[-1].valid else observations[-1].reason)
        self.update_buttons()
    def update_buttons(self):
        self.start_btn.setEnabled(self.worker is None);self.cal_btn.setEnabled(self.worker is not None and self.worker.frame_size is not None)
        self.track_btn.setEnabled(self.stage.session is not None and self.stage.session.state=='results-view')
    def begin(self):
        if not self.worker:return
        self.config=self.configuration();legacy=self.comparison[self.blocks.__len__()] if self.comparison else False
        self.stage.start(self.worker.run_id,self.screen_selected(),legacy)
    def results(self,export):
        export['config']=self.config
        try:atomic_json(DATA/'last-validation.json',{**export['report'],'config':self.config})
        except (OSError,ValueError) as e:self.status.setText(f'Result available; could not save summary: {e}')
        self.status.setText(self.stage.result_text());self.update_buttons()
        if self.comparison:self.blocks.append(export)
    def track_or_next(self):
        if not self.stage.session or self.stage.session.state!='results-view':return
        if self.comparison:
            if len(self.blocks)<4:self.begin()
            else:self.comparison=None;self.status.setText('Comparison collected. Export diagnostics to analyze the four blocks.')
        else:self.stage.track()
    def quick(self):
        if self.stage.tracker:self.stage.quick_check()
    def toggle_hold(self):
        self.stage.manual_capture=not self.stage.manual_capture;self.status.setText('Hold H during calibration captures.' if self.stage.manual_capture else 'Automatic stable capture enabled.')
    def posture(self):
        if not self.worker or self.worker.framing!='off':self.status.setText('Disable auto-framing and establish camera identity before the geometry experiment.');return
        self.stage.posture_pack()
    def cancel(self):self.stage.cancel();self.comparison=None
    def compare(self):
        if not self.worker:return
        if self.camera_profile:self.status.setText('Restart the app without importing intrinsics before comparing the original method.');return
        self.comparison=[True,False,False,True];self.blocks=[];self.begin()
    def save(self):
        if not self.stage.tracker or not self.stage.report or self.stage.session.state not in ('results-view','tracking'):return
        path,_=QFileDialog.getSaveFileName(self,'Save calibration',str(DATA/'calibration.json'),'JSON (*.json)')
        if path:
            try:save_profile(path,self.stage.tracker.mapper,self.config,self.stage.tracker.guard.to_list(),self.stage.report,self.stage.tracker.offset)
            except Exception as e:self.status.setText(str(e))
    def load(self):
        if not self.worker:return
        path,_=QFileDialog.getOpenFileName(self,'Load calibration',str(DATA),'JSON (*.json)')
        if path:
            try:
                self.config=self.configuration();m,d=load_profile(path,self.config);self.stage.setGeometry(self.screen_selected().geometry());self.stage.quick_check(self.worker.run_id,m,d)
            except Exception as e:self.stage.cancel();self.status.setText(str(e))
    def export(self):
        if not self.stage.export:return
        path,_=QFileDialog.getSaveFileName(self,'Export numeric diagnostics',str(DATA/'session.json'),'JSON (*.json)')
        if path:
            try:atomic_json(path,dict(schema=2,synthetic=False,blocks=self.blocks) if len(self.blocks)==4 else self.stage.export)
            except Exception as e:self.status.setText(str(e))
    def import_camera(self):
        if self.worker:self.status.setText('Stop the camera before importing intrinsics.');return
        path,_=QFileDialog.getOpenFileName(self,'Import camera intrinsics',str(DATA),'JSON (*.json)')
        if path:
            try:
                from calibrate_camera import validate_profile
                self.camera_profile=validate_profile(read_json(path));self.status.setText('Intrinsics loaded. Start camera and recalibrate.')
            except Exception as e:self.status.setText(str(e))
    def stop(self):
        self.stage.cancel();self.comparison=None
        if self.worker:self.worker.requestInterruption()
        self.status.setText('Stopping camera…' if self.worker else 'Camera is off.');self.preview.clear();self.update_buttons()
    def finished(self):self.worker=None;self.stage.cancel();self.update_buttons()
    def closeEvent(self,event):
        self.stop()
        if self.worker or self.stage.fitters:event.ignore();return
        self.stage.close();event.accept()

if __name__=='__main__':
    app=QApplication(sys.argv);app.setApplicationName('GazeTorch Demo');window=Control();window.show();sys.exit(app.exec())
