import numpy as np
from pypylon import pylon



def connect_camera():
    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice()
    )
    camera.Open()
    return camera

def camera_settings(camera):
    camera.PixelFormat.SetValue("Mono12")
    camera.ExposureTime.SetValue(150.0)
    camera.Gain.SetValue(0.0)
    camera.AcquisitionMode.SetValue("Continuous")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return camera

def disconnect_camera(camera):
    camera.StopGrabbing()
    # camera.ExposureTime.SetValue(150.0)
    camera.Close()

def acquire_data(camera):
    res = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    img = res.Array
    w, h = res.Width, res.Height
    t = res.GetTimeStamp() / 1e9
    res.Release()
    return img, w, h, t

def acquire_time_series(camera, Trec, rect, dA, K):
    i0, i1, j0, j1 = rect
    T = list()
    I = list()
    _, _, _, t0 = acquire_data(camera)
    Tmax = t0 + Trec
    t = 0
    while t < Tmax:
        X, _, _, t = acquire_data(camera)
        T.append(t)
        I.append(X)
    T = np.array(T) - min(T)
    I = np.array([K * integrate_area(x[i0:i1, j0:j1] / 4095, dA) for x in I])
    return T, I

