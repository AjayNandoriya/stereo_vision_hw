import cv2
import pytest

@pytest.mark.skip(reason="only test locally")
def test_cam_available():
    cam = cv2.VideoCapture(0)
    assert cam.isOpened()
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,240)    
    ret_frame, frame = cam.read()
    cam.release()
    assert ret_frame
    assert frame.shape[0] == 240
    assert frame.shape[1] == 320

def test_default():
    assert 1==1

if __name__ == '__main__':
    test_cam_available()