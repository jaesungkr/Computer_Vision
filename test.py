import numpy as np, cv2


def drawFlow(img, flow, step=16):
    h, w = img.shape[:2]
    # 16픽셀 간격의 그리드 인덱스 구하기
    idx_y, idx_x = np.mgrid[step / 2:h:step, step / 2:w:step].astype(np.int)
    indices = np.stack((idx_x, idx_y), axis=-1).reshape(-1, 2)
    for x, y in indices:  # 인덱스 순회
        # 각 그리드 인덱스 위치에 점 그리기
        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        # 각 그리드 인덱스에 해당하는 플로 결과 값(이동 거리)
        dx, dy = flow[y, x].astype(np.int)
        # 각 그리드 인덱스 위치에서 이동한 거리만큼 선 그리기
        cv2.line(img, (x, y), (x + dx, y + dy), (0, 255, 0), 2, cv2.LINE_AA)


prev = None  # 이전 프레임 저장 변수

cap = cv2.VideoCapture('./videos/mine.mp4')

delay = int(1000 / 15)  # delay = 1000 / 15fps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 최초 프레임의 경우
    if prev is None:
        prev = gray  # 첫 이전 프레임
    else:
        # 이전, 이후 프레임으로 옵티컬 플로 계산
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.1,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        # 계산 결과 그리기, 선언한 함수 호출
        drawFlow(frame, flow)
        # 다음 프레임을 위해 이월
        prev = gray

    cv2.imshow('OpticalFlow-Farneback', frame)

    if cv2.waitKey(delay) == 27:
        break

cap.release()

cv2.destroyWindow()