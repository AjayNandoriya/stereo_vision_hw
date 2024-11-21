import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob


def calibrate():
    
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob('*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    # calibration 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 
    img = cv2.imread('left12.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # 
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)


    #
    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)

    # reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )

    pass


def test_():
    mtx = np.eye(3)
    dist = (0,0,0,0,0,0)

    w,h = 256,256
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    pass



def initUndistortRectifyMap(mtx, dist, size):
    u,v = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    return undistortPoints(u,v, mtx, dist)

def undistortPoints(u,v, mtx, dist):
    fx = mtx[0,0]
    fy = mtx[1,1]
    cx = mtx[0,2]
    cy = mtx[1,2]
    k4,k5,k6 = 0,0,0
    k1,k2,p1,p2,k3 =dist
    x = (u-cx)/fx
    y = (v-cy)/fy

    # R
    # x_1,y_1 = (R1*X)/(R3*X), (R2*X)/(R3*X)
    x_1,y_1 = x,y


    x2 = x_1**2
    y2 = y_1**2
    r2 = x2 + y2
    
    r4 = r2*r2
    r6 = r2*r4

    dist_k_top = 1 + k1*r2 + k2*r4 + k3*r6
    dist_k_bot = 1 + k4*r2 + k5*r4 + k6*r6
    dist_k = dist_k_top/dist_k_bot
    x_2 = x_1*(dist_k) + 2*p1*x_1*y_1 + p2*(r2 + 2*x2)
    y_2 = y_1*(dist_k) + p1*(r2 + 2*y2) + 2*p2*x_1*y_1

    # R
    # x_3,y_3 = (invR1*X)/(invR3*X), (invR2*X)/(invR3*X)
    x_3,y_3 = x_2, y_2


    out_x = x_3*fx + cx
    out_y = y_3*fy + cy
    return out_x, out_y


def test_initUndistortRectifyMap():
    h,w = 256,256
    fx,fy = 10,0.5
    cx,cy = w//2, h//2
    k1,k2,k3 = 0.0005,0.0000005,0.0000001
    p1,p2 = 0.0001,0.0001
    mtx = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    dist = (k1,k2,p1,p2,k3)
    mapx_gt,mapy_gt = cv2.initUndistortRectifyMap(mtx, dist,None, mtx, (w,h), 5)

    mapx, mapy = initUndistortRectifyMap(mtx, dist, (w,h))

    vmax = 0.5
    fig,axs = plt.subplots(2,3,sharex=True, sharey=True)
    axs[0,0].imshow(mapx, vmin=0,vmax=w)
    axs[0,1].imshow(mapx_gt, vmin=0,vmax=w)
    axs[0,2].imshow(mapx-mapx_gt, vmin=-vmax, vmax=vmax)
    axs[1,0].imshow(mapy, vmin=0,vmax=h)
    axs[1,1].imshow(mapy_gt, vmin=0,vmax=h)
    axs[1,2].imshow(mapy-mapy_gt, vmin=-vmax, vmax=vmax)
    plt.show()

def rodrigue(src):
    src = src.reshape((3,1))
    theta = np.linalg.norm(src)
    if abs(theta)<0.00001:
        src_normed = src*0.0
    else:    
        src_normed = src/theta
    I = np.eye(3)
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    rx,ry,rz = src_normed[:,0]
    rrt = np.matmul(src_normed, src_normed.T)
    Ri = np.array([[0,-rz,-ry],[rz,0,-rx],[-ry,rx,0]], np.float32)
    R = cos_th*I + (1-cos_th)*rrt + sin_th*Ri
    return R
    
def test_rodrigue():
    rvec = np.array([1,0,0],dtype=np.float32)
    R_gt,_ = cv2.Rodrigues(rvec)
    R = rodrigue(rvec)
    plt.plot(R_gt.reshape(-1),label='gt')
    plt.plot(R.reshape(-1),label='out')
    plt.plot(R.reshape(-1)-R_gt.reshape(-1),label='diff')
    plt.grid(),plt.legend()
    plt.show()
    pass

def projectPoints(points3d, rvec, tvec, mtx, dist):
    N=points3d.shape[0]
    points3d_d = []

    tvec = tvec.reshape((1,3))
    rvec = rvec.reshape((1,3))
    # R,_ = cv2.Rodrigues(rvec)
    R = rodrigue(rvec)
        
    p3 = points3d
    points3d_d = np.matmul(p3, R.T) + tvec

    points2d = points3d_d/points3d_d[:,2:3]
    points2d = np.matmul(points2d,mtx)
    points2d = points2d[:,:2]
    points2d = points2d[:,np.newaxis,:]
    return points2d

def test_projectPoints():
    N = 100
    objpoints = np.random.random((N,3))*100
    rvec = np.array([0.1,0.1,1], dtype=np.float32)
    tvec = np.array([10,0,0], dtype=np.float32)
    mtx = np.eye(3)
    dist = (0,0,0,0,0)
    imgpoints2_gt, _ = cv2.projectPoints(objpoints, rvec, tvec, mtx, dist)

    imgpoints2 = projectPoints(objpoints, rvec, tvec, mtx, dist)

    diff = imgpoints2 - imgpoints2_gt
    plt.plot(np.linalg.norm(imgpoints2_gt[:,0,:],axis=1), label='gt')
    plt.plot(np.linalg.norm(imgpoints2[:,0,:],axis=1), label='out')
    plt.plot(np.linalg.norm(diff[:,0,:],axis=1), label='diff')
    plt.grid(), plt.legend()
    plt.show()
        
if __name__ == '__main__':
    # test_()
    # test_initUndistortRectifyMap()
    test_projectPoints()
    # test_rodrigue()
