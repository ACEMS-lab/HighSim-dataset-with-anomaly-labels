"""
gmm_outside.py —— GMM "类外判断"。

GMM 的 argmax 总会把一个点分进某一类，即使它离所有类都很远。原来的做法直接用该类的 UCL，
于是 z≥5 的急刹车会被分进训练数据里的"急加减速类"（z 约 2–5），用最宽松的 UCL，被合理化。

现在：
  1. 每个窗口仍取后验概率最大的类 k；
  2. 计算它到类 k 中心的马氏距离² d²；与类 k 训练窗口 d² 的 P{OUTSIDE_Q} 比较；
  3. 在类内 → 用类 k 的 UCL；在类外（不像任何一种正常驾驶）→ 用最严格（最小）的 UCL。
"""
import numpy as np

OUTSIDE_Q = 0.99


def _precisions_full(gmm):
    P = gmm.precisions_
    k, d = gmm.means_.shape
    ct = gmm.covariance_type
    if ct == "full":
        return P
    if ct == "tied":
        return np.repeat(P[None], k, axis=0)
    if ct == "diag":
        return np.stack([np.diag(p) for p in P])
    return np.stack([np.eye(d) * p for p in P])          # spherical


def mahalanobis2(gmm, Z, labels):
    """每个样本到其所属类中心的马氏距离²"""
    P = _precisions_full(gmm)
    out = np.empty(len(Z))
    for k in np.unique(labels):
        m = labels == k
        diff = Z[m] - gmm.means_[k]
        out[m] = np.einsum("ij,jk,ik->i", diff, P[k], diff)
    return out


def fit_outside_thresholds(gmm, Z_train, q=OUTSIDE_Q):
    """每类的 d² 阈值 = 该类训练样本 d² 的 q 分位数"""
    lab = gmm.predict(Z_train)
    d2 = mahalanobis2(gmm, Z_train, lab)
    thr = np.full(gmm.n_components, np.inf)
    for k in range(gmm.n_components):
        if (lab == k).any():
            thr[k] = float(np.quantile(d2[lab == k], q))
    return thr


def window_ucl(gmm, Z, cluster_ucl, d2_thr):
    """返回 (每个窗口的 UCL, 是否类外, 所属类)"""
    lab = gmm.predict(Z)
    d2 = mahalanobis2(gmm, Z, lab)
    ucl_v = np.array([cluster_ucl[k] for k in range(gmm.n_components)], float)
    outside = d2 > d2_thr[lab]
    ucl = np.where(outside, ucl_v.min(), ucl_v[lab])
    return ucl, outside, lab
