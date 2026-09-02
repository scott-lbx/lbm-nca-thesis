"""風向角度轉換。

模型永遠以「風從左（+x）進」的 canonical 方向推論。使用者在 Rhino 框選任意角度的
boundary 後，`src/core/geometry_ingest.py::ingest_analyze` 用本模組把 world 座標的
幾何旋轉進 boundary 自身的座標系（長邊 → +x），再做 bbox 正規化。

## 座標契約（唯一真相）

``ingest_analyze`` 產出的 ``polygons`` **一律已是 canonical 方向**（風沿 +x）的
``[0, 1]`` 正規化座標——兩套方言皆然：

* web 扁平 sim-px 方言：``wind_angle`` 恆為 0，本來就 canonical。
* GH world 多邊形方言：在此模組旋轉 −θ 後才正規化。

因此 ``src/ggui/window.py`` **不得**再對 polygons 做第二次旋轉。先前的實作是
「world AABB 正規化 → window 再轉 −θ」，兩步都對，合起來錯：旋轉後 boundary 的
axis-aligned bounding box 比 boundary 本身大且長寬比不同，x/y 被不同倍率壓縮，
接著在這個非等向縮放的空間裡旋轉會把建物切變（shear）成平行四邊形。

## 為什麼沒有「速度場轉回世界座標」

GGUI 顯示的是 canonical frame：mask、建物、流場三者同在風沿 +x 的座標系，彼此一致。
把速度向量單獨轉回 world 角度會讓「向量方向」與「像素位置」分屬兩個座標系，比不轉
更錯——真要以 Rhino 方位顯示，必須連整張影像一起旋轉，那是另一個功能。
本模組因此不提供速度場逆旋轉。
"""
import math


class AngleTransform:
    """把 world 座標的點旋轉進 canonical 方向（風沿 +x）。"""

    def __init__(self, angle_deg: float):
        """初始化。

        Args:
            angle_deg: boundary 最長邊相對 +x 軸的角度（度，CCW 為正）。
        """
        self.angle_deg = angle_deg
        rad = math.radians(-angle_deg)          # 轉至 canonical = 旋轉 −θ
        self._cos = math.cos(rad)
        self._sin = math.sin(rad)

    def to_canonical(
        self,
        points: list[list[float]],
        center: tuple[float, float] = (0.0, 0.0),
    ) -> list[list[float]]:
        """旋轉點列至 canonical 方向（CPU 操作，每次幾何變更一次）。

        ``center`` 只影響平移量；呼叫端接著都會做 bbox 正規化把平移消掉，
        故用哪個中心不影響最終結果，預設原點即可。

        Args:
            points: ``[[x, y], ...]`` world 座標。
            center: 旋轉中心。

        Returns:
            旋轉後的 ``[[x, y], ...]``。
        """
        cx, cy = center
        cos_a, sin_a = self._cos, self._sin
        out = []
        for (x, y) in points:
            dx, dy = x - cx, y - cy
            out.append([dx * cos_a - dy * sin_a + cx, dx * sin_a + dy * cos_a + cy])
        return out

    @staticmethod
    def no_transform() -> "AngleTransform":
        """角度為 0 的 identity transform（風向正對左側，無需旋轉）。"""
        return AngleTransform(0.0)
