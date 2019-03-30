import cv2
import numpy as np


def can_detect(img, lower_color, upper_color):
    # ブラー処理。誤検出を減らす。
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    # HSV空間に変換
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 指定した色範囲で2値化
    threash = cv2.inRange(hsv, lower_color, upper_color)

    # 輪郭検出
    contours, hierarchy = cv2.findContours(threash,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:  # 輪郭が検出できたら
        contours.sort(key=cv2.contourArea, reverse=True)  # 輪郭による面積が大きい順にソート
        contour = contours[0]  # 一番大きのを取り出す

        # 検出した輪郭を近似。
        # contourを描画すると分かるが、めちゃくちゃ細かく輪郭を取ってる。
        # つまり点の数が多い。これだと処理の負担が大きい。
        # その負担を減らすために近似する。
        # みたいな感じ。ざっくりしすぎだから解釈違い起こすかもしれんが、まー詳しくは調べて。
        arclen = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour,
                                  0.01 * arclen,
                                  True)

        # 検出した輪郭の各頂点の座標から、缶を表す四角形の頂点４つを抽出

        # X座標でソート
        approx_x = approx[np.argsort(approx[:, 0, 0])[::-1]]  # X座標を基準に降順でソート
        cord_x_max = np.array(approx_x[0, 0])  # X座標が一番大きいやつ
        cord_x_min = np.array(approx_x[-1, 0])  # X座標が一番小さいやつ

        # Y座標でソート
        approx_y = approx[np.argsort(approx[:, 0, 1])[::-1]]  # X座標を基準に降順でソート
        cord_y_max = np.array(approx_y[0, 0])  # X座標が一番大きいやつ
        cord_y_min = np.array(approx_y[-1, 0])  # X座標が一番小さいやつ

        # 検出した輪郭(近似版)を描画する。
        # 近似する前のを描画したいなら、[approx]をcontourに変えるだけ。
        # _img = cv2.drawContours(img, [approx], -1, (0, 255, 0), 5)

        # 抽出した頂点４つをひとまとめに
        points = np.vstack([cord_x_min, cord_x_max, cord_y_min, cord_y_max])
        points = points[np.argsort(points[:, 0])]  # X座標を基準に、小さい順にソート

        # 検出した点の、３つめと４つめを入れ替える
        # cv2.lineで線引くときに分かりやすくする(連続性をもたせる)ため
        _p = points[2].copy()
        points[2] = points[3]
        points[3] = _p

        return points
    else:
        return None


if __name__ == "__main__":
    img = cv2.imread("can_pics/sample1.jpg", 1)
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
