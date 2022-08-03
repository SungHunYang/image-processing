from pathlib import Path
from networks.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import os
import torchsummary


##### 필요에 따라 수정

# 아직 사용 x
NON_INDEXED_IMAGE_PATH = Path("./data/nonindex")
NON_INDEXED_IMAGE_PATH.mkdir(parents=True, exist_ok=True)

# 모델과 출력 위치 결정
MODEL_PATH = Path("./models/rosa_mb2-ssd-lite-Epoch-299-Loss-1.4975405500994787-3000.pth")
OUTPUT_PATH = Path('./data/output')
OUTPUT_PATH.mkdir(parents=True,
                  exist_ok=True)  # parents = True인 경우 상위 디렉토리를 포함하여 경로의 누락 된 디렉토리를 작성, 만일 이게 False면 누락이 있으면 FileNotFoundError # exist_ok=True 경로에 폴더가 없는 경우 자동 생성

# 가져올 이미지와 csv 파일 경로
IMAGE_PATH = Path("./data/test_real")
DataFrame = pd.read_csv("./data/test_black_cat_dog_long.csv")

# mAP 측정을 위한 이전 값들 저장
MODEL = ['0720_test', '0725_test','0801_test','Rosa_test','before_test','0802_test']
Before_mAP = [40.0, 43.0, 52.3, 54.8, 85.1,43.6] # 단 Rosa님 test는 30, before일때는 20 IOU_THRESHOLD

# 모델에 들어간 클래스 순서에 맞게 수정
# CLASS_NAMES = ["BACKGROUND","BlackDog","CatFace","DogFace","LongDogFace"] # 0720 # 0802
CLASS_NAMES = ["BACKGROUND","Dogface","BlackDog","CatFace"] # rosa님 모델
# CLASS_NAMES = ["BACKGROUND","DogFace","LongDogFace"] # 0725
# CLASS_NAMES = ["BACKGROUND", "CatFace", "DogFace", "LongDogFace"] # 0801
# CLASS_NAMES = ["BACKGROUND", "CatFace", "DogFace"] # before
# CLASS_NAMES = ["BACKGROUND","DogNose"]

# 원하는 THRESHOLD 값 저장
# THRESHOLDS = [n for n in range(50, 100, 5)]  # 50 ~ 95 까지 일단 만들고
IOU_THRESHOLD = 50  # 얼만큼 예측 박스와 truth 박스의 차이가 나야 잘 나왔다고 할 지
Config_THRESHOLD = 90  # 얼만큼 이상의 confidence를 잘 찾았다고 할지

##### 거의 수정 불필요
NETWORK = create_mobilenetv2_ssd_lite(len(CLASS_NAMES), is_test=True)  # return이 SSD 즉, NETWORK는 SSD임
NETWORK.load(MODEL_PATH)  # 이미 모델이 있는 경우 load
torchsummary.summary(NETWORK,input_size = (3,300,300)) # 모델 정보 추출

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
predictor = create_mobilenetv2_ssd_lite_predictor(NETWORK, candidate_size=200, device=device)
# candidate size 늘리면 cpu를 더 많이 먹고 더 많은 사진을 가져올 듯


def count_class(DataFrame) -> dict:  # 모든class len check
    class_len = {}
    check = DataFrame['ClassName'].unique()
    for u in check:
        ans = DataFrame['ClassName'] == u
        counts = ans.value_counts()[1]
        class_len[u] = counts

    return class_len


def get_iou(gt_box: tuple, box: tuple) -> float:
    gt_box_left = gt_box[0]
    gt_box_top = gt_box[1]
    gt_box_right = gt_box[2]
    gt_box_bottom = gt_box[3]

    A_gt = (gt_box_right - gt_box_left) * (gt_box_bottom - gt_box_top)

    pred_box_left = box[0]
    pred_box_top = box[1]
    pred_box_right = box[2]
    pred_box_bottom = box[3]

    A_pred = (pred_box_right - pred_box_left) * (pred_box_bottom - pred_box_top)

    # 왼쪽 위 point
    x1 = np.maximum(pred_box_left, gt_box_left)
    y1 = np.maximum(pred_box_top, gt_box_top)

    # 오른쪽 아래 point
    x2 = np.minimum(pred_box_right, gt_box_right)
    y2 = np.minimum(pred_box_bottom, gt_box_bottom)

    # 교차 영역 구하기
    A_intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    A_union = A_gt + A_pred - A_intersection
    iou = abs(A_intersection / A_union)

    return iou


def make_pr_curve(recall_list: list, precision_list: list, class_name: str, th: int, iou_th :int , ap: float) -> None:
    plt.figure(figsize=(9, 9))
    plt.xlabel("Recall")
    plt.xlim(0, 1)
    plt.title(f"PR Curve \n {class_name} \n threshold : 0.{th} , iou_threshold : 0.{iou_th} \n ap : {ap:.5f}")
    plt.ylim(0, 1)
    plt.ylabel("Precision")
    plt.scatter(recall_list, precision_list, c="b", s=20)  # s 는 마커의 크기
    plt.plot(recall_list, precision_list)  # "rs--" 라인 스타일을 점선으로

    plt.show()


def calc_ap(recall_list: list,
            precision_list: list) -> float:  # Precision 이랑 recall이랑 정렬해야 하지 않나? 정렬하고 , 거기에 맞게 넓이를 구해야 할것 같은데,일단 대기
    ap = 0
    for i in range(len(precision_list)):
        if i == 0:
            continue
        else:
            ap += (recall_list[i] - recall_list[i - 1]) * precision_list[i]

    return ap


def draw_box(image, image_id: str, box: tuple, gt_box: tuple, label: str, confidence: float, THRESHOLD: int,
             ClassName: str) -> None:
    output_image = f"{OUTPUT_PATH}/{'SUCCESS'}/{ClassName}"
    OUT_success = Path(output_image)
    OUT_success.mkdir(parents=True, exist_ok=True)

    output_image_s = f"{OUTPUT_PATH}/{'FAIL'}/{ClassName}"
    OUT_fail = Path(output_image_s)
    OUT_fail.mkdir(parents=True, exist_ok=True)

    if box[0] > 1000:  # 사진 size가 1000보다 크면인가?
        line_weight = 5
    elif 100 < box[0] <= 1000:
        line_weight = 2
    else:
        line_weight = 1

    cv2.rectangle(
        image,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        (255, 0, 255),
        line_weight
    )

    cv2.rectangle(
        image,
        (int(gt_box[0]), int(gt_box[1])),
        (int(gt_box[2]), int(gt_box[3])),
        (0, 255, 255),
        line_weight
    )
    text = f"{CLASS_NAMES[label]} : {confidence:.3f}"

    cv2.putText(image, text,
                (
                    int(box[0] + 20), int(box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
                )
    # cv2.putText(image, text,
    #             (
    #                 int(box[0]), int(box[1]) ),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.4,
    #             (255, 0, 255),
    #             1
    #             ) # nose 일때 글씨좀 작게 보려고
    if confidence * 100 > THRESHOLD:
        output_image_path = f"{OUT_success}/{image_id}"
        cv2.imwrite(output_image_path, image)
    else:
        output_image_path = f"{OUT_fail}/{image_id}"
        cv2.imwrite(output_image_path, image)


def unpredict_draw_box(image, image_id: str, box: tuple, gt_box: tuple, label: str, confidence: float,
                       gt_label: str) -> None:
    output_image_n = f"{OUTPUT_PATH}/{'Not Found'}"
    OUT_not = Path(output_image_n)
    OUT_not.mkdir(parents=True, exist_ok=True)

    if box[0] > 1000:  # 사진 size가 1000보다 크면인가?
        line_weight = 5
    elif 100 < box[0] <= 1000:
        line_weight = 3
    else:
        line_weight = 2

    cv2.rectangle(
        image,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        (255, 0, 255),
        line_weight
    )

    cv2.rectangle(
        image,
        (int(gt_box[0]), int(gt_box[1])),
        (int(gt_box[2]), int(gt_box[3])),
        (0, 255, 255),
        line_weight
    )
    text = f"{gt_label}->{CLASS_NAMES[label]} : {confidence:.3f}"

    cv2.putText(image, text,
                (
                    int(box[0] + 10), int(box[1]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,

                (255, 0, 255),
                2
                )

    output_image_path = f"{OUT_not}/{image_id}"
    cv2.imwrite(output_image_path, image)

def undraw_box(image, image_id: str, gt_box: tuple,  ClassName: str) -> None:

    un_image_s = f"{OUTPUT_PATH}/{'UnPredict'}"
    OUT_un = Path(un_image_s)
    OUT_un.mkdir(parents=True, exist_ok=True)

    if gt_box[0] > 1000:  # 사진 size가 1000보다 크면인가?
        line_weight = 5
    elif 100 < gt_box[0] <= 1000:
        line_weight = 2
    else:
        line_weight = 1

    cv2.rectangle(
        image,
        (int(gt_box[0]), int(gt_box[1])),
        (int(gt_box[2]), int(gt_box[3])),
        (0, 255, 255),
        line_weight
    )
    text = f"{ClassName} : Not Predict"

    cv2.putText(image, text,
                (
                    int(gt_box[0] + 20), int(gt_box[1]) + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2
                )

    output_image_path = f"{OUT_un}/{image_id}"
    cv2.imwrite(output_image_path, image)


def compare_iou_threshold(iou: float, confidence : float ,iot_th: int,confi_th : int ,accTP, accFP):
    if iou * 100 > iot_th:
        # True Positive
        if confidence*100 > confi_th:
            return accTP + 1, accFP
        else:
            return accTP, accFP
    else:
        # False Positive
        if confidence * 100 > confi_th:
            return accTP, accFP + 1
        else:
            return accTP, accFP


def calc_pr(accTP: int, accFP: int,gt_length: int) -> tuple:
    if accTP == 0 and accFP == 0:
        return 0, 0
    else:
        recall = accTP / gt_length
        precision = accTP / (accTP + accFP)
        return precision, recall


def mAP(ap_list: list, class_len: dict, model: list = None, before_mAP: list = None) -> None:
    # 갑자기 드는 생각인데, mAP를 측정하고 싶으면 nms이전에 측정해야 하는게 맞는거 아닌가?

    mAP = sum(ap_list)
    mAP /= len(class_len)
    mAP = round(mAP, 3)

    if model is None:
        y = [100, mAP * 100]
        x = ['perfect', 'result']
        plt.figure(figsize=(9, 9))
        plt.xlabel("model")
        plt.ylabel("mAP_value")
        plt.ylim(0, 100)
        plt.title(f"mAP")
        plt.bar(x, y, width=0.5)
        for i, v in enumerate(x):
            plt.text(v, y[i], str(y[i]) + "%",  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
                     fontsize=9,
                     color='red',
                     horizontalalignment='center',  # horizontalalignment (left, center, right)
                     verticalalignment='bottom')

    else:
        before_mAP.append(mAP * 100)
        plt.figure(figsize=(9, 9))
        plt.xlabel("model")
        plt.ylabel("mAP_value")
        plt.title(f"mAP")
        plt.bar(model, before_mAP, width=0.5)
        for i, v in enumerate(model):
            plt.text(v, before_mAP[i], str(before_mAP[i]) + "%",  # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
                     fontsize=9,
                     color='red',
                     horizontalalignment='center',  # horizontalalignment (left, center, right)
                     verticalalignment='bottom')

    plt.show()


image_data = list()
unimage_data = list()
recall_list = list()
precision_list = list()
image_data_dict = dict()
Data = count_class(DataFrame)

for class_name, count in Data.items():

    images = glob.glob(f"./{IMAGE_PATH}/{class_name.lower()}/*.jpg")  # 여기조건에 맞는 사진들이 필요
    image_data = []

    for image in images:
        orig_image = cv2.imread(image)
        cvt_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 순으로 바꾸는 듯
        boxes, labels, confidences = predictor.predict(cvt_image, 10, 0.1)

        # 하나씩 보려고 하는 구나
        image_id = os.path.basename(image)  # 사진 이름만 출력
        image_info = DataFrame[DataFrame["ImageID"] == image_id]

        try:
            gt_xmin = float(image_info.iat[0, 1])
            gt_ymin = float(image_info.iat[0, 2])
            gt_xmax = float(image_info.iat[0, 3])
            gt_ymax = float(image_info.iat[0, 4])
            gt_label = str(image_info.iat[0, 5])
            gt_box = (gt_xmin, gt_ymin, gt_xmax, gt_ymax)
            gt_length = count

        except IndexError:
            continue

        data = list()  # 매번 초기화

        for idx in range(len(boxes)):  # 선택된 박스 길이
            if CLASS_NAMES[labels[idx]] == class_name:  # 박스에 맞는 라벨과, 내가 선택한 클래스가 같은지
                data.append([boxes[idx], labels[idx], confidences[idx]])

        data.sort(key=lambda x: x[2], reverse=True)  # 정확도 순으로 내림차순

        if len(data) == 0:
            undraw_box(orig_image, image_id, gt_box, class_name)
            continue
        else:
            box, label, confidence = data[0]  # 클래스에 맞는 가장 잘 예측한 값
            pred_info = (image_id, box, label, confidence, gt_box, class_name)
            image_data.append(pred_info)

        if CLASS_NAMES[label] != gt_label:  # csv에 정한 class와 맞는지 아닌지
            unpredict_draw_box(orig_image, image_id, box, gt_box, label, confidence, gt_label)
        else:
            draw_box(orig_image, image_id, box, gt_box, label, confidence, Config_THRESHOLD, class_name)

    image_data_dict[class_name] = image_data
    image_data_dict[class_name].sort(key=lambda x: x[3], reverse=True)

ap_list = []

for class_name, c_data in image_data_dict.items():
    length = Data[class_name]
    ap = 0
    accTP = 0
    accFP = 0

    for data in c_data:
        image_id = data[0]
        box = data[1]
        label = data[2]
        confidence = data[3]
        gt_box = data[4]
        gt_classname = class_name

        iou = get_iou(gt_box, box)
        accTP, accFP = compare_iou_threshold(iou,confidence,IOU_THRESHOLD,Config_THRESHOLD,
                                                                     accTP, accFP)
        precision_list.append(calc_pr(accTP, accFP, length)[0])
        recall_list.append(calc_pr(accTP, accFP,length)[1])

    ap = calc_ap(recall_list, precision_list)
    ap_list.append(ap)
    make_pr_curve(recall_list, precision_list, gt_classname, Config_THRESHOLD,IOU_THRESHOLD,ap)
    recall_list.clear()
    precision_list.clear()  # 새로 할당 하는것 보다는 지우는게 속도에 더 좋겠지?

mAP(ap_list, Data, MODEL, Before_mAP)  # 이전 모델들의 mAP 비교를 위함
# mAP(ap_list, Data)  # 이번에 구한 mAP만 확인하고 싶을 때
