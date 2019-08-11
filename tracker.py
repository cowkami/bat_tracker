from time import sleep
import numpy as np
import cv2



VIDEO_DIR  = '/home/lee/bat_research/data/raw/190526/teshima movie/20190525/'
BAT_NUMS   = ['bat290', 'bat294', 'bat296', 'bat298']
TRIAL_NUMS = ['no1', 'no2', 'no3', 'no4', 'no6']

VIDEO_PATHS = []
for b in BAT_NUMS:
    for t in TRIAL_NUMS:
        for rl in [2, 3]:
            VIDEO_PATHS.append(
                f'{VIDEO_DIR}/{b}/{t}/{t}_001/{t}_001_NX8-S1 Camera(0{rl})/NX8-S1 Camera.avi',
            )

for i, path in enumerate(VIDEO_PATHS):
    print(i, path)



def read_frame(cap):
    _, frame = cap.read()
    return frame


def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std


def convert_img_8bits(image):
    mx = image.max()
    mn = image.min()
    return (255 * ((image - mn) / (mx - mn))).astype('uint8')


def derivative_2frames(frame1, frame2):
    f1, f2 = np.array(frame1), np.array(frame2)
    out = (f2[:, :] - f1[:, :])**2
    #out = normalize(out)
    #out = cv2.GaussianBlur(out, (25, 25), 2)
    #out = np.where(out<-0.2, out.max(), out.min())
    out = convert_img_8bits(out)
    return out


def optical_flow(frame1, frame2):
    f1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    flow = cv2.calcOpticalFlowFarneback(
        f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


def cal_mean_frame(path_to_video, start=None, stop=0, frame_rate=50):
    cap = cv2.VideoCapture(path_to_video)
    mean_frame = read_frame(cap)
    frames = []
    frame_num = 0
    for i in range(100):
        read_frame(cap)
        frame_num += 1

    while cap.isOpened():
        frame = read_frame(cap)
        frame_num += 1
        if frame_num >= 150:
            break
        frames.append(frame)
        mean_frame += frame

    while True:
        cv2.imshow('mean frame', convert_img_8bits(mean_frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def play(path_to_video, start=None, stop=0, frame_rate=50):
    '''
    play video.

    path_to_video: str
    start: float or int (time(sec) from trigger)
    stop: float or int (time(sec) from trigger)
    '''
    cap = cv2.VideoCapture(path_to_video) 
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_length = frame_count / frame_rate
    if start is None:
        start = -video_length

    frame_num = 0
    for i in range(130):
        read_frame(cap)
        frame_num += 1

    frame1 = read_frame(cap)
    frame_num += 1

    while cap.isOpened():
        frame2 = read_frame(cap)
        if (
            cv2.waitKey(1) & 0xFF == ord('q') or
            frame2 is None or
            frame_num >= 300
        ):
            break

        frame_sum = np.zeros(frame1.shape)
        display = np.append(
            #derivative_2frames(frame1, frame2),
            optical_flow(frame1, frame2),
            convert_img_8bits(normalize(frame2)),
            axis=0
        )
        cv2.imshow('frame', display)
        frame1 = np.copy(frame2)
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video = []
    for i in range(frame_count):
        _, frame = cap.read()
        video.append(frame)
    cap.release()
    return np.array(video)


def play_video(video):
    flag = True
    while flag:
        for frame in video:
            cv2.imshow('frame', frame)
            sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                flag = False
                break
    cv2.destroyAllWindows()


def convert_video2feature(video, feature_func):
    feature = []
    for i in range(video.shape[0]-1):
        feature.append(feature_func(video[i], video[i+1]))
    return np.array(feature)


def test_func(frame1, frame2):
    opf = cv2.cvtColor(optical_flow(frame1, frame2), cv2.COLOR_BGR2GRAY)
    drv = cv2.cvtColor(derivative_2frames(frame1, frame2), cv2.COLOR_BGR2GRAY)
    out = normalize(opf*drv)
    out = cv2.GaussianBlur(out, (25, 25), 2)
    out = convert_img_8bits(out)
    return out


def main():
#    play(VIDEO_PATHS[31])
    #cal_mean_frame(VIDEO_PATHS[30])
    v = load_video(VIDEO_PATHS[30])
    f = convert_video2feature(v, test_func)
    play_video(f)


if __name__ == '__main__':
    main()
