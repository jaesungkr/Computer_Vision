import numpy as np, cv2

vid = cv2.VideoCapture('./videos/mine.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (int(vid.get(3)), int(vid.get(4))))
delay = int(1000 / 15)  # slows the video
# random color_forLines only for lines
flag = 0
color_forLines = np.random.randint(0, 255, (200, 3))
lines = None
previous = None  # previous img
blackscreen = 0 # black screen variable(button)
#feature paramter for goodfeaturestotrack function
feature_params = dict( maxCorners = 30,qualityLevel = 0.01,minDistance = 10,blockSize = 8)

# calcOpticalFlowPyrLK termination
termi = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.08)

while True:
    ret, frame = vid.read()
    if ret == False:
        exit()
    Drawing = frame.copy()
    #change to gray img
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # if first image
    if previous is None:  #if theres no image before, meaning first
        previous = gray
        # image frame with np.zeros
        lines = np.zeros_like(frame)
        # find corner
        mask = np.zeros_like(previous)
        mask[:] = 300
        prevPt = cv2.goodFeaturesToTrack(previous,mask=mask, **feature_params)

    else:  #after the first image
        nextImg = gray
        # opticalflow to search next point and calculate
        nextPt, status, err = cv2.calcOpticalFlowPyrLK(previous, nextImg, prevPt, None, criteria=termi)
        # find moved points
        #delete 0 from 0,1 for good points
        ##print(prevPt)
        pre_mov = prevPt[status != 0]
        nex_mov = nextPt[status != 0]
        if blackscreen == 1:
            Drawing = np.zeros((int(vid.get(4)), int(vid.get(3)), 3), np.uint8)

        nofPoints = int(pre_mov.size / 2) # which equals to maxCorners
        prex_sum,prey_sum, nexx_sum, nexy_sum = 0,0,0,0
        for i in range(nofPoints):
            prex_sum = prex_sum + pre_mov[i][0]
            prey_sum = prey_sum + pre_mov[i][1]
            nexx_sum = nexx_sum + nex_mov[i][0]
            nexy_sum = nexy_sum + nex_mov[i][1]
        av_pre_x = prex_sum / nofPoints
        av_pre_y = prey_sum / nofPoints
        av_nex_x = nexx_sum / nofPoints
        av_nex_y = nexy_sum / nofPoints
        #average move
        av_mov_x = abs(av_pre_x - av_nex_x)
        av_mov_y = abs(av_pre_y - av_nex_y)
        for i, (p, n) in enumerate(zip(pre_mov, nex_mov)):
            px, py = p.reshape(-1)
            nx, ny = n.reshape(-1)
            # draw line between previous and current points
            cv2.circle(Drawing, (int(nx), int(ny)), 2, (0, 255, 0), -1)
            if av_mov_x > av_mov_y:
                if abs(px-nx)-0.9 < av_mov_x:
                    continue
                else:
                    cv2.line(lines, (int(px), int(py)), (int(nx), int(ny)), color_forLines[i].tolist(), 2)
                    # plot circle on new points
            elif av_mov_x < av_mov_y:
                if abs(py-ny)-0.9 < av_mov_y:
                    continue
                else:
                    cv2.line(lines, (int(px), int(py)), (int(nx), int(ny)), color_forLines[i].tolist(), 2)
                    # plot circle on new points
            else:
                continue


           # cv2.line(lines, (int(px), int(py)), (int(nx), int(ny)), color_forLines[i].tolist(), 2)
            # plot circle on new points
            #cv2.circle(Drawing, (int(nx), int(ny)), 2, (0, 255, 0), -1)

        # add to img-draw
        Drawing = cv2.addWeighted(Drawing, 0.8, lines, 0.3, 0.1)
        # store current img to previous variable

        previous = nextImg
        prevPt = nex_mov.reshape(-1, 1, 2)

    #draw
    cv2.imshow('Jaesung Park', Drawing)
    #recording
    writer.write(Drawing)
    key = cv2.waitKey(delay)
    #esc button to quit
    if key == 27:
        break
    elif key == ord('b'): # b to screen black
        if blackscreen == 0:
            blackscreen = 1
        else:
            blackscreen = 0
    elif key == ord('d'): # d to delete previous lines for more clear view
        previous = None

cv2.destroyAllWindows()
vid.release()
writer.release()
