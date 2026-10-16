# GazeTorch webcam dot for Mac

## Setup

Use Python 3.11 on macOS. From the repository root:

    python3.11 -m venv .venv
    .venv/bin/python -m pip install -r demo/requirements.txt
    mkdir -p demo/assets

Download `GazeTorch_MobileNet.pkl` from the project's [official checkpoints](https://huggingface.co/7thChord/GazeTorch/tree/main) into `demo/assets/`. This checkpoint contains Python objects and is loaded with `weights_only=False`, so use the trusted project checkpoint, not an arbitrary pickle.

Download MediaPipe's [canonical face model](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj) into `demo/assets/canonical_face_model.obj`. Keep the filename unchanged. Models and images are not included in this contribution.

Build the local Mac launcher, then open it:

    .venv/bin/python demo/build_macos_app.py
    open 'GazeTorch Demo.app'

The generated bundle points to your checkout and virtualenv. Rebuild it at a new output path if you move either. The builder refuses to overwrite an existing bundle. macOS may ask for camera access on first use. You can also run `.venv/bin/python demo/app.py` directly, although camera permission will belong to the launching Python/terminal context.

## Use

Open **GazeTorch Demo.app**, then choose **Start camera → Calibrate**. Look at the small center of each target, keeping your head comfortably still. Use even lighting and disable Center Stage or other auto-framing.

Calibration visits 13 locations twice, including near the screen edges. It waits for a stable fixation and resets after blinks, tracking loss, stale observations, or excessive movement. Repeated visits must agree. A short preflight estimates eye openness and feature noise. Expect roughly a minute or longer if targets need retries.

Nine separate targets then measure accuracy. The results distinguish raw prediction error from displayed-dot error, show per-target errors, and report usable time. **Press Enter** at the result screen to show the dot. Error is in Qt logical screen points, not physical pixels or degrees. Webcam accuracy may still be too coarse for small controls.

- **D:** desktop overlay. This is visual only and intercepts clicks.
- **C:** fresh calibration.
- **R:** retry a target after automatic retries are exhausted.
- **Esc:** stop and release the camera.

The dot disappears on stale data or unreliable posture. Return to your calibrated posture, use Quick check, or recalibrate.

## Advanced

- **Hold-to-capture:** optionally hold **H** when ready at each calibration target. The same stability rules apply.
- **Quick check:** five stable proposal visits and five independent confirmation visits. Only a small consistent translation is accepted. An incomplete check or position-dependent error requires recalibration.
- **Save / Load calibration:** JSON only. Loading checks camera, display, source/model, framing, and intrinsics identity, then requires Quick check. Unknown display identity can prevent reuse even on a seemingly unchanged setup. Saved files are never overwritten by a failed check.
- **Export diagnostics:** explicitly saves numerical observations, targets, timing, and results. No images or video are exported. Routine summaries contain no full observation series.
- **Run comparison:** original/new then new/original blocks with fresh calibration and the same nine-position validation. Press Enter between blocks, then export all four. This requires your participation; synthetic checks do not prove improved gaze accuracy.
- **Posture calibration experiment:** after a full calibration, collect 20 additional visits at small left/right/near/far offsets. A fixed geometry candidate is compared against the current mapper on fresh posture and still-head targets. It is retained only if the error and availability gates pass; otherwise the simpler mapper and its original posture limits remain. A final independent validation follows either result. This uses approximate face geometry, not a measured eyeball origin.
- **Import camera intrinsics:** stop the camera first. Profiles must match the selected camera and delivered resolution. Importing changes preprocessing and requires fresh gaze calibration. Restart the app to return to approximate intrinsics.

Optional physical camera calibration, with a real checkerboard of 9×6 inner corners:

    .venv/bin/python demo/calibrate_camera.py --camera 0 --columns 9 --rows 6 --square-mm 25 --output demo/evaluation/camera-intrinsics.json

Replace 25 with the measured square size in millimeters. Space captures a view; Esc cancels. Capture at least 20 diverse views with tilts about both axes, varied distance, and coverage near image edges, then five fresh check views. Front-facing-only boards can give misleadingly low residuals and are rejected. Frames stay in memory. No physical intrinsics profile has been verified merely by shipping this helper.

## Local verification

    QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s demo/tests -p 'test_*.py'
    .venv/bin/python demo/test_camera_worker.py
    .venv/bin/python demo/engine.py
    .venv/bin/python demo/evaluate_calibration.py --synthetic --output demo/evaluation/synthetic-report.json
    .venv/bin/python demo/evaluate_calibration.py --session demo/evaluation/human-session.json --output demo/evaluation/human-comparison.json

For the worker regression, place a consented face image at `demo/test-assets/face.jpg` (or set `GAZETORCH_TEST_FACE` to its path). This image stays local and is not included in the repository. The worker regression uses it with real landmarks/model inference and three start/stop cycles. It does not activate the camera. The app uses CPU inference and a 16 MiB QThread stack to preserve the native crash fix. Fitting also runs off the UI thread.

Runtime summaries and optional profiles default to `~/Library/Application Support/GazeTorch Demo/`. Older target-median summaries are not comparable to the new frame-level metrics. OpenCV supplies software retrieval timing only; sensor exposure timing and backend frame loss remain unknown.

The native bundle loads Python from this checkout, so restart it after source changes. `Launch GazeTorch.command` is the direct Python launcher; use the native bundle for the normal Mac permission experience.
