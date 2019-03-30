from ball_detect import search_largest_circle
from can_detect import can_detect

import cv2
import os

if __name__ == "__main__":
    """ ボール検出 """
    print(os.path.exists("samples/ball1.jpg"))
    img = cv2.imread("samples/ball1.jpg", 1)  # RBGカラーで読み込み
    x, y, r, label = search_largest_circle(img,  # 入力画像
                                           (0, 150, 0), (5, 255, 255),  # 赤のHSVでのレンジ
                                           (100, 150, 0), (180, 255, 255),  # 青のHSVでのレンジ
                                           (20, 150, 0), (40, 255, 255)  # 黄のHSVでのレンジ
                                           )

    print(x, y, r, label)

    # 検出した円をマーク
    cv2.circle(img, (x, y), r, (0, 255, 0), 5)

    cv2.namedWindow("res", cv2.WINDOW_NORMAL)  # ウィンドウサイズ可変に設定
    cv2.imshow("res", img)  # 検出結果表示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """ 缶検出 """
    img = cv2.imread("samples/can1.jpg", 1)
    points = can_detect(img, (10, 100, 100), (40, 255, 255))
    if points is not None:
        cv2.line(img, tuple(points[0]), tuple(points[1]), (255, 150, 0), 10)
        cv2.line(img, tuple(points[1]), tuple(points[2]), (255, 150, 0), 10)
        cv2.line(img, tuple(points[2]), tuple(points[3]), (255, 150, 0), 10)
        cv2.line(img, tuple(points[3]), tuple(points[0]), (255, 150, 0), 10)

        cv2.namedWindow("res", cv2.WINDOW_NORMAL)
        cv2.imshow("res", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Can not found")
