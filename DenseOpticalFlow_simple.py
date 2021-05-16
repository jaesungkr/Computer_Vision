import numpy as np, cv2

vid = cv2.VideoCapture('./videos/third.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('DenseOpticalFlow.avi', fourcc, 30.0, (int(vid.get(3)), int(vid.get(4))))
prev = None
delay = int(1000 / 60)  # delay = 60fps make it faster since its originally slow to calculate
# step = 16 = pixels
def dense(img, flow, step=16):
    height, width = img.shape[:2]
    # Find grid index of 16 pixels
    store_y, store_x = np.mgrid[step/2:height:step, step/2:width:step].reshape(2, -1).astype(int)
    indices = np.stack((store_x, store_y), axis=-1).reshape(-1, 2)
    for x, y in indices:
        # circle on each grid
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        # distance
        dx, dy = flow[y, x].astype(np.int)
        # line on flows
        cv2.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # first frame
    if prev is None:
        prev = gray
    else:
        # using opticalflowfarneback
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        dense(frame, flow)
        #renew
        prev = gray

    cv2.imshow('Dense', frame)
    writer.write(frame)
    #black backscreen is not applied in this dense implementation.
    if cv2.waitKey(delay) == 27:
        break

vid.release()
writer.release()
cv2.destroyAllWindows()