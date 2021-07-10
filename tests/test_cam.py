import cv2

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


def test_feature_extraction():
    cam = cv2.VideoCapture(0)
    assert cam.isOpened()
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,240)    
    ret_frame, frame = cam.read()
    
    sift = cv2.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray, kp, img)

if __name__ == '__main__':
    test_cam_available()