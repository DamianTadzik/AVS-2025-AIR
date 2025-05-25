# from google.colab import drive
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
import imutils
import os
from os.path import join
import time

# drive.mount('/content/gdrive')
DATASET_DIR = './sequences/'

SIGMA = 17
SEARCH_REGION_SCALE = 2
LR = 0.125
NUM_PRETRAIN = 128
VISUALIZE = True

# %matplotlib inline
import matplotlib.pyplot as plt
def cv2_imshow(cv2image):
    plt.imshow(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
    plt.show()

from IPython.display import display, clear_output
def jupyter_imshow(img, title="img"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    clear_output(wait=True)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def load_gt(gt_file):

    with open(gt_file, 'r') as file:
        lines = file.readlines()

    lines = [line.split(',') for line in lines]
    lines = [[int(float(coord)) for coord in line] for line in lines]
    # returns in x1y1wh format
    return lines


def show_sequence(sequence_dir, title='img'):

    imgdir = join(sequence_dir, 'color')
    imgnames = os.listdir(imgdir)                  
    imgnames.sort()
    gt_boxes = load_gt(join(sequence_dir, 'groundtruth.txt'))

    for imgname, gt in zip(imgnames, gt_boxes):
        img = cv2.imread(join(imgdir, imgname))
        position = [int(x) for x in gt]
        cv2.rectangle(img, (position[0], position[1]), (position[0]+position[2], position[1]+position[3]), (255, 0, 0), 2)
        # cv2.imshow('demo', img)
        # if cv2.waitKey(0) == ord('q'):
        #     break
        jupyter_imshow(img, title)
        time.sleep(0.01)


def crop_search_window(bbox, frame):
    print(f"{bbox=}")

    if len(bbox) == 4:
        xmin, ymin, w, h = bbox
    else:
        raise ValueError(f"Expected bbox of length 4, got {len(bbox)}: {bbox}")

    cx = xmin + w / 2
    cy = ymin + h / 2

    #---TODO (1): Zwiększ okno wyszukiwania
    w_search = int(w * SEARCH_REGION_SCALE)
    h_search = int(h * SEARCH_REGION_SCALE)

    xmin = int(cx - w_search / 2)
    xmax = int(cx + w_search / 2)
    ymin = int(cy - h_search / 2)
    ymax = int(cy + h_search / 2)

    #---TODO (2): Padding
    x_pad = max(0, -xmin, xmax - frame.shape[1])
    y_pad = max(0, -ymin, ymax - frame.shape[0])

    # Dodaj odbicie ramki jeśli okno wychodzi poza obraz
    if x_pad > 0 or y_pad > 0:
        frame = cv2.copyMakeBorder(frame, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_REFLECT)
        xmin += x_pad
        xmax += x_pad
        ymin += y_pad
        ymax += y_pad

    # Wycięcie i konwersja na grayscale
    window = frame[ymin:ymax, xmin:xmax]
    # window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

    return window



def get_gauss_response(gt_box):

    width = gt_box[2] * SEARCH_REGION_SCALE
    height = gt_box[3] * SEARCH_REGION_SCALE
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    center_x = width // 2
    center_y = height // 2
    dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * SIGMA)
    response = np.exp(-dist)

    return response

def pre_process(img, special_treatment=False):

    if special_treatment:
        height, width, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        height, width = img.shape
    img = img.astype(np.float32)

    #---- TODO (3)
    img = img + 1
    img = np.log(img)
    img = img - np.mean(img)
    img = img / np.std(img)
    #---- TODO (3)

    #2d Hanning window
    win_col = np.hanning(width)
    win_row = np.hanning(height)
    mask_col, mask_row = np.meshgrid(win_col, win_row)
    window = mask_col * mask_row
    img = img * window

    return img

def random_warp(img):

    #---TODO (4)
    angle = np.random.uniform(-15, 15)
    img_rot = imutils.rotate_bound(img, angle)
    img_resized = cv2.resize(img_rot, (img.shape[1], img.shape[0]))
    #---TODO (4)

    return img_resized


def initialize(init_frame, init_gt):

    g = get_gauss_response(init_gt)
    G = np.fft.fft2(g)
    Ai, Bi = pre_training(init_gt, init_frame, G)

    return Ai, Bi, G


def pre_training(init_gt, init_frame, G):

    template = crop_search_window(init_gt, init_frame)
    fi = pre_process(template)
    print(f"{G.shape=}")
    print(f"{np.conjugate(np.fft.fft2(fi)).shape=}")
    
    Ai = G * np.conjugate(np.fft.fft2(fi))                # (1a)
    Bi = np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi))  # (1b)

    for _ in range(NUM_PRETRAIN):
        fi = pre_process(random_warp(template))

        Ai = Ai + G * np.conjugate(np.fft.fft2(fi))               # (1a)
        Bi = Bi + np.fft.fft2(fi) * np.conjugate(np.fft.fft2(fi)) # (1b)

    return Ai, Bi


def track(image, position, Ai, Bi, G):
    x, y, w, h = position
    response = predict(image, position, Ai / Bi)
    position = update_position(response, position, w, h)
    Ai, Bi = update(image, position, Ai, Bi, G)
    return position, Ai, Bi



def predict(frame, position, H):

    #----TODO (5)
    # Wytnij okno z obrazu
    x = crop_search_window(position, frame)
    
    # Wstępne przetwarzanie  
    x = pre_process(x)
    
    # Transformacja do dziedziny częstotliwości
    F = np.fft.fft2(x)
    
    # Odpowiedź filtra w dziedzinie częstotliwości
    response_freq = H * F
    
    # Powrót do dziedziny przestrzennej
    gi = np.real(np.fft.ifft2(response_freq))
    #----TODO (5)
    
    return gi


def update(frame, position, Ai, Bi, G):

    #----TODO (5)
    # Wytnij okno z obrazu
    x = crop_search_window(frame, position)

    # Wstępne przetwarzanie
    x = pre_process(x)

    # Transformacja do dziedziny częstotliwości
    F = np.fft.fft2(x)

    # Aktualizacja Ai i Bi
    Ai_new = G * np.conj(F)
    Bi_new = F * np.conj(F)

    # Aktualizacja z uwzględnieniem współczynnika uczenia
    Ai = (1 - LR) * Ai + LR * Ai_new
    Bi = (1 - LR) * Bi + LR * Bi_new
    #----TODO (5)

    return Ai, Bi


def update_position(spatial_response, position, w, h):
    # Znajdź współrzędne maksimum odpowiedzi
    dy, dx = np.unravel_index(np.argmax(spatial_response), spatial_response.shape)

    # Oblicz przesunięcie względem środka
    center_y, center_x = spatial_response.shape[0] // 2, spatial_response.shape[1] // 2
    shift_y = dy - center_y
    shift_x = dx - center_x

    # Zaktualizuj pozycję obiektu (tylko x, y)
    new_x = int(position[0] + shift_x)
    new_y = int(position[1] + shift_y)

    return [new_x, new_y, w, h]


def bbox_iou(box1, box2):

    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[0], box1[0] + box1[2]
    b1_y1, b1_y2 = box1[1], box1[1] + box1[3]
    b2_x1, b2_x2 = box2[0], box2[0] + box2[2]
    b2_y1, b2_y2 = box2[1], box2[1] + box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * np.clip(inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def test_sequence(DATASET_DIR, sequence):
    seqdir = join(DATASET_DIR, sequence)
    imgdir = join(seqdir, 'color')
    imgnames = sorted(os.listdir(imgdir))

    init_img = cv2.imread(join(imgdir, imgnames[0]))
    gt_boxes = load_gt(join(seqdir, 'groundtruth.txt'))
    init_gt = gt_boxes[0]
    init_gray = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    
    Ai, Bi, G = initialize(init_gray, init_gt)

    results = [init_gt]

    # for image in images
    for imgname in imgnames[1:]:
        img = cv2.imread(join(imgdir, imgname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        position, Ai, Bi = track(gray, results[-1], Ai, Bi, G)
        results.append(position)

        if VISUALIZE:
            draw_pos = [int(i) for i in position]
            cv2.rectangle(img, (draw_pos[0], draw_pos[1]), (draw_pos[0]+draw_pos[2], draw_pos[1]+draw_pos[3]), (255, 0, 0), 2)
            cv2_imshow(img)

    return results, gt_boxes



# if __name__ == "__main__":
#     sequences = ['jump']
#     ious_per_sequence = {}
#     for sequence in sequences:

#         results, gt_boxes = test_sequence(DATASET_DIR, sequence)
#         ious = []
#         for res_box, gt_box in zip(results, gt_boxes[1:]):
#             iou = bbox_iou(res_box, gt_box)
#             ious.append(iou)

#         ious_per_sequence[sequence] = np.mean(ious)
#         print(sequence, ':', np.mean(ious))

#     print('Mean IoU:', np.mean(list(ious_per_sequence.values())))

if __name__ == "__main__":
    sequences = ['jump']
    ious_per_sequence = {}

    for sequence in sequences:
    

        results, gt_boxes = test_sequence(DATASET_DIR, sequence)
        ious = [bbox_iou(pred, gt) for pred, gt in zip(results[1:], gt_boxes[1:])]
        mean_iou = np.mean(ious)

        ious_per_sequence[sequence] = mean_iou
        print(f"{sequence}: {mean_iou:.3f}")

    print('Mean IoU:', np.mean(list(ious_per_sequence.values())))
