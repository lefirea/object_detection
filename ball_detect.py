import cv2
import numpy as np


def getCircle(hsv, lower_color, upper_color, min_r=80, max_r=130):
    # ブラー処理。誤検出が減る
    blur = cv2.GaussianBlur(hsv, (9, 9), 0)

    # 2値化。今回はカラー画像なので、スレッショルドではない。
    # この関数は、色の範囲(下限と上限)を指定して、その範囲だけを１(白)に、他を０(黒)にする。
    color = cv2.inRange(blur, lower_color, upper_color)

    # ノイズ除去。やらなくても大丈夫。
    # ネット上の解説ページだと、やってるのもやってないのもあった。どんだけ効果があるかは分からん。
    # 速さが欲しいならコメントアウトするくらいで良いんじゃないかな。精度がどんだけ変わるか分からんが。
    element8 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.uint8)
    oc = cv2.morphologyEx(color, cv2.MORPH_OPEN, element8)
    oc = cv2.morphologyEx(oc, cv2.MORPH_CLOSE, element8)

    # 輪郭検出。OpenCV3系だと返り値が３つだったんだが、openCV4系になって２つになった。
    # OpenCV3系を使ってる人は、これを丸写ししないこと。
    contours, hierarchy = cv2.findContours(oc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("{} contours".format(len(contours)))

    # 見つけたなかで一番妥当な円を探す
    if len(contours) > 0:  # そもそも輪郭が検出できたか
        contours.sort(key=cv2.contourArea, reverse=True)  # 検出した輪郭による領域を基準に、大きい順にソート
        for i in range(len(contours)):
            cnt = contours[i]  # 大きいものから順に取り出す

            # 最小外接円を使って円を検出
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))

            if min_r <= radius <= max_r:  # 検出した半径が、許容最小値、許容最大値の範囲に収まっていれば
                return center, radius  # 中心座標と半径を返す

        return None, None  # 望む円が検出できなければNoneを返す
    else:
        return None, None  # 望む円が検出できなければNoneを返す


def search_largest_circle(frame,
                          r_hsv_min, r_hsv_max,
                          b_hsv_min, b_hsv_max,
                          y_hsv_min, y_hsv_max):
    # グレー空間に変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # HSV空間に変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    r_max = 0  # 半径の最大値
    x = 0  # 中心座標Xの値
    y = 0  # 中心座標Yの値
    label = 0  # 色情報。other=0, red=1, blue=2, yellow=3

    # 赤、青、黄のボールをそれぞれ検出
    center_r, r_r = getCircle(hsv, r_hsv_min, r_hsv_max)
    center_b, r_b = getCircle(hsv, b_hsv_min, b_hsv_max)
    center_y, r_y = getCircle(hsv, y_hsv_min, y_hsv_max)

    """ 半径の大きさを比較 """
    """ 効率の良い方法が思いつかないので愚直にRed,Blue,Yellow全パターンの組み合わせを比較する """
    # Red と Blue の比較
    if (r_r is not None) and (r_b is not None):  # どっちもNoneじゃない
        r_max = max(r_max, r_r, r_b)
        if r_max == r_r:  # Redが一番大きいなら
            x = center_r[0]
            y = center_r[1]
            label = 1
        elif r_max == r_b:  # Blueが一番大きいなら
            x = center_b[0]
            y = center_b[1]
            label = 2
    elif (r_b is None) and (r_r is not None):  # BlueだけNoneなら
        r_max = max(r_max, r_r)
        if r_max == r_r:
            x = center_r[0]
            y = center_r[1]
            label = 1
    elif (r_r is not None) and (r_b is not None):  # RedだけNoneなら
        r_max = max(r_max, r_b)
        if r_max == r_b:
            x = center_b[0]
            y = center_b[1]
            label = 2

    # Blue と Yellow の比較
    if (r_b is not None) and (r_y is not None):  # どっちもNoneじゃない
        r_max = max(r_max, r_b, r_y)
        if r_max == r_b:  # Blueが一番大きいなら
            x = center_b[0]
            y = center_b[1]
            label = 2
        elif r_max == r_y:  # Yellowが一番大きいなら
            x = center_y[0]
            y = center_y[1]
            label = 3
    elif (r_y is None) and (r_b is not None):  # YellowだけNoneなら
        r_max = max(r_max, r_b)
        if r_max == r_b:
            x = center_b[0]
            y = center_b[1]
            label = 2
    elif (r_b is None) and (r_y is not None):  # BlueだけNoneなら
        r_max = max(r_max, r_y)
        if r_max == r_y:
            x = center_y[0]
            y = center_y[1]
            label = 3

    # Yellow と Red の比較
    if (r_y is not None) and (r_r is not None):  # どっちもNoneじゃない
        r_max = max(r_max, r_y, r_r)
        if r_max == r_y:  # Yellowが一番大きいなら
            x = center_y[0]
            y = center_y[1]
            label = 3
        elif r_max == r_r:  # Redが一番大きいなら
            x = center_r[0]
            y = center_r[1]
            label = 1
    elif (r_r is None) and (r_y is not None):  # RedだけNoneなら
        r_max = max(r_max, r_y)
        if r_max == r_y:
            x = center_y[0]
            y = center_y[1]
            label = 3
    elif (r_y is None) and (r_r is not None):  # YellowだけNoneなら
        r_max = max(r_max, r_r)
        if r_max == r_r:
            x = center_r[0]
            y = center_r[1]
            label = 1

    x = int(x)
    y = int(y)
    r_max = int(r_max)
    return x, y, r_max, label  # 検出したボールの情報を返す


if __name__ == "__main__":
    img = cv2.imread("/samples/ball1.jpg", 1)  # RBGカラーで読み込み
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
