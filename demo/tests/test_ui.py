import os,sys,unittest,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtWidgets import QApplication
from app import Stage,Control,FitWorker
from calibration import *
from tracking import Tracker
from test_calibration import obs

class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_full_synthetic_session_and_fitter(self):
        s=Session('r',(1000,1000));clock=0;sequence=0
        while s.state!='fitting':
            if s.present_pending:s.presented(clock)
            target=s.target;f=(target[0]-.5,target[1]-.5,0,0)
            sequence+=1;clock+=.05
            s.receive(Observation('r',sequence,clock,clock,features=f,openness=(.2,.2),pose=(0,0,0,0,0,600)),clock);s.advance(clock)
            self.assertLess(clock,100)
        self.assertEqual(len(s.visits),26)
        worker=FitWorker(s);results=[]
        from PySide6.QtCore import Qt
        worker.result.connect(lambda *args:results.append(args),Qt.ConnectionType.DirectConnection);worker.start();self.assertTrue(worker.wait(10000));self.assertEqual(results[0][-1],'')
        mapper=results[0][1];s.begin_validation(mapper)
        while s.state!='results':
            if s.present_pending:s.presented(clock)
            target=s.target;clock+=.05;sequence+=1
            s.receive(Observation('r',sequence,clock,clock,features=(target[0]-.5,target[1]-.5,0,0)),clock);s.advance(clock)
        from evaluate_calibration import score_validation
        report,_,_=score_validation(s.validation,mapper,s.size)
        self.assertTrue(report['raw']['complete']);self.assertLess(report['raw']['mean_error'],3)
    def test_cancel_stale_fit_and_controls(self):
        control=Control();control.show();self.app.processEvents()
        self.assertFalse(control.cal_btn.isEnabled());self.assertTrue(control.start_btn.isEnabled())
        stage=control.stage;stage.start('r',self.app.primaryScreen());old=stage.session.id
        stage.cancel();stage.fit_result(old,None,[], 'old error');self.assertIsNone(stage.session)
        control.close();stage.timer.stop();control.timer.stop()
    def test_delivery_silence_hides_dot(self):
        mapper=Mapper('gaze').fit([[0,0],[1,0],[0,1]],[[0,0],[1,0],[0,1]],robust=False)
        tracker=Tracker(mapper,(1000,1000));tracker.receive(obs(1),1);self.assertIsNotNone(tracker.dot)
        tracker.advance(1.3);self.assertIsNone(tracker.dot)

class AdvancedFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def test_origin_asymmetric_and_rotated(self):
        import cv2
        from engine import normalization_origin
        face=np.zeros((468,3));face[[33,133,362,263]]=[2,0,0];face[[98,327]]=[10,0,0]
        np.testing.assert_allclose(normalization_origin(face,np.zeros(3),[0,0,600]),[6,0,600])
        np.testing.assert_allclose(normalization_origin(face,np.array([0,0,np.pi/2]),[0,0,600]),[0,6,600],atol=1e-9)
    def test_quick_proposal_confirmation_and_cancel(self):
        stage=Stage();stage.setGeometry(0,0,1000,1000);stage.session=Session('r',(1000,1000))
        mapper=Mapper('gaze',.001).fit([[0,0],[1,0],[0,1],[1,1]],[[0,0],[1,0],[0,1],[1,1]],robust=False)
        stage.tracker=Tracker(mapper,(1000,1000));stage.quick_check();clock=0;sequence=0
        for _ in range(1000):
            s=stage.session
            if s.present_pending:s.presented(clock)
            clock+=.05;sequence+=1;x,y=s.target
            s.receive(Observation('r',sequence,clock,clock,features=(x-.02,y-.01,0,0)),clock);s.advance(clock)
            if s.state=='quick-proposal-ready':stage.finish_quick_proposal()
            if s.state=='results':stage.finish_validation();break
        self.assertEqual(stage.session.state,'results-view');self.assertIsNone(stage.quick_phase)
        np.testing.assert_allclose(stage.tracker.offset,[20,10],atol=2)
        self.assertLess(stage.report['raw']['mean_error'],2)
        stage.cancel();self.assertIsNotNone(stage.last_validated);stage.timer.stop();stage.close()
