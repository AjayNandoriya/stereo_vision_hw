import os
import cv2
import numpy as np

class StereoCam(object):
    def __init__(self):
        cam_ids = [1,2]
        self.cams =[]
        for cam_id in cam_ids:
            cam = cv2.VideoCapture(cam_id)
            if not cam.isOpened():
                print("Cannot open camera {0}".format(cam_id))
                continue
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,320)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
            self.cams.append(cam)

    def __del__(self):
        for cam in self.cams:
            cam.release()
        
    def get_images(self):
        images = []
        ret = True
        for icam, cam in enumerate(self.cams):
            # Capture frame-by-frame
            ret_frame, frame = cam.read()
            # if frame is read correctly ret is True
            if not ret_frame:
                print("Can't receive frame: {}".format(icam))
                images.append(None)
                ret = False
            else:
                images.append(frame)
        return ret, images

    def calibrate(self):
        # get image-pairs
        # get feature pairs
        
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
            objectPoints, imagePoints1, imagePoints2,
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
            imageSize, flags=cv2.CALIB_FIX_INTRINSIC)

def get_sharpness(img):
    img_gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_lap_abs = np.abs(cv2.Laplacian(img_gray, -1, ksize=3))
    mask = img_lap_abs>10
    img_lap_val = np.sum(img_lap_abs)/np.sum(mask)
    return img_lap_val


def test_stereo_cam():
    from matplotlib import pyplot as plt
    steree_cam = StereoCam()
    while(1):
        ret, images = steree_cam.get_images()
        if all([img is None for img in images]):
            break
        # frame = np.concatenate(images,axis=1)
        for iframe, frame in enumerate(images):
            if frame is not None:
                sharpness = get_sharpness(frame)
                cv2.putText(frame, 'sharpness:{0:.2 f}'.format(sharpness), (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,255,255), thickness=1)
                cv2.imshow('frame_{0:d}'.format(iframe), frame)
                print(frame.shape)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    test_stereo_cam()
