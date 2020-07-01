# Opencv
---
- Opencvで読み込まれる画像はNumPyのArrayとして扱うことができる
-----
## 画像の表示
```python
import cv2

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)         # ウィンドウサイズの変換が手動で行えるようになる
cv2.imshow(window_name, img)

while True:
    k = cv2.waitKey(0)
    if k == ord("q"):               # この場合、qを押すとwhile から抜けだす
        break

cv2.destroyWindow(window_name)        # windowを削除
cv2.destroyAllWindows()                        # すべてのwindowを削除
```
---
## 色の変換
- opencvの場合、色の取り扱いはB、G、Rの順番で行われる
```python
import cv2

# BGR -> GRAY
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# GRAY -> BGR
bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
```
---
## 任意のフィルタの作成及び画像への適用
```python
import cv2
import numpy as np

def generate_kernel():
    # numpy 形式で任意のサイズのフィルタを作成
    # 5x5のメジアンフィルタの場合
    kernel = np.ones((5, 5), np.float32) / (5*5)
    return kernel
    
def apply_filter(img, kernel):
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img
```
---
## 画像の差分の計算方法
- 基本的にはuint8で扱われるため、そのまま引き算すると0以下の値が他の値に設定されるため、float32等に型変換する必要がある

```python
import cv2
import numpy as np

float_img_1 = uint8_img_1.astype(np.float32)
float_img_2 = uint8_img_2.astype(np.float32)

float_diff_img = float_img_2 - float_img_1
float_diff_img[diff_img<0] = 0        # 0以下の要素をすべて0に設定する
uint8_diff_img = float_diff_img.astype(np.uint8)
```
---
## 各図形の表示
```python
import cv2

# 線分
cv2.line(img, (x_1, y_1), (x_2, y_2), color, thickness, lineType)
# 円
cv2.circle(img, (x_o, x_y))
```
---
## Contourの作成
- Contourを利用することで、画像内の図形の輪郭抽出を行うことができる
- 検出された輪郭は階層構造(hierarchy)で菅るされる
- 詳しくは[輪郭の階層情報]([輪郭の階層情報 — OpenCV-Python Tutorials 1 documentation](http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_contours/py_contours_hierarchy/py_contours_hierarchy.html) "輪郭抽出")を参照

### contourの取得
```python
import cv2

def get_contours(img):
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)    # bgr -> gray

    ret, thresh_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)       
    # 第２引数で閾値を設定
    # 第４引数でどのような変化を持たせるかを指定(この例の場合0または255になる)
    # thresh_img : 閾値処理が施された結果

    img, contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours : 各IDの属する輪郭点
    # hierarchy : 各輪郭の親子構造が保存される
    
    return contours, hierarchy
```

### 最大図形の抽出
```python
import cv2

# 親となる図形のみを抽出
def get_parent(contours, hierarchy):
    contours_copy = contours.copy()         # [1, n]
    hierarchy_copy = hierarchy.copy()        # np.ndarray
    length = len(contours)              # 図形の数を取得

    count = 0         # いくつ図形を削除したかを記憶
    for i in range(length):
        if not hierarchy[0][i][-1] == -1:       # hierarchy[0][i][-1]が-1の時、その図形は親がいない図形である
            del contours_copy[i-count]
            np.delete(hierarchy[0], i-count, 0)
            count += 1

    # 親となる図形の情報のみを保存
    contours = contours_copy()          
    hierarchy = hierarchy_copy()
    
    return contours, hierarchy
    
def find_max_area(contours):
    max_area = 0
    max_index = 0
    length = len(contours)
    
    for i in range(length):
        area = cv2.contourArea(contours[i])
        
        if area > max_area:
            max_area = area
            max_index = i
    
    return max_index
    
contours, hierarchy = get_contours(img)
parent_contours, parent_hierarchy = get_parent(contours, hierarchy)
max_index = find_max_area(parent_contours)
```
---
## マウスクリックによる操作
- このクラスを利用した場合、一瞬クリックしただけれも複数回分のクリックとして認識されるため、前回クリックした場所の座標を覚えておくことで、それと比較するなどの対応が必要である
```python
class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)
    
    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType    
        self.mouseEvent["flags"] = flags    

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent
    
    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]                

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]                

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]  

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]  

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])
        
    cv2.imshow(window_name, img)
    mouse = mouseParam(window_name)
    
    while True:
        if mouse.getEvent() == cv2.EVENT_LBUTTONDOWN:
            # 左クリックされたときの動作
        elif mouse.getEvent() == cv2.EVENT_RBUTTONDOWN:
            # 右クリックされたときの動作
        elif mouse.getEvent() == cv2,EVENT_MBUTTONDOWN:
            # ホイールが押されたときの動作
        
        x = mouse.getX()      # クリックした位置のX座標を取得
        y = mouse.getY()      # クリックした位置のY座標を取得
```

---
## 特定の色の部分のみを抽出
- BGRから[HSV](https://ja.wikipedia.org/wiki/HSV%E8%89%B2%E7%A9%BA%E9%96%93)へ変換した場合
```python
import cv2
import numpy as np

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([x_l, y_l, z_l])
upper = np.array([x_u, y_u, z_u])
img_mask = cv2.inRange(hsv, lower, upper)
img_color = cv2.bitwise_and(img, img, mask=img_mask)

```

---
## テキストの表示
```python
import cv2

cv2.putText(img, text, (x, y), font, size, (b, g, r), width)
```

--- 
## 特徴点取得
- ORB特徴量の場合
```python
import cv2

orb_extractor = cv2.ORB_create()
kp, des = orb_extractor.detectAndCompute(img, mask=None)

# 特徴点の座標を見る場合
print(kp.pt)
```
