import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import label
from skimage.morphology import skeletonize_3d

def estimate_local_curvature(coords, k=points):
    curvatures = []
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)

    for pt in coords:
        _, indices = nbrs.kneighbors([pt])
        neighborhood = coords[indices[0]]
        pca = PCA(n_components=3)
        pca.fit(neighborhood)
        eigvals = pca.explained_variance_ + 1e-8
        curvature = eigvals[2] / eigvals.sum()
        curvatures.append(curvature)

    return np.array(curvatures)

def split_pancreas_mask_by_skeleton(total_mask, com_mask):
    #스켈레톤화 
    skeleton = skeletonize_3d(com_mask)
    coords = np.argwhere(skeleton > 0)
    if len(coords) < 30:
        print("정보 부족")
        return None

    #곡률 계산
    curvatures = estimate_local_curvature(coords)

    #head-body, body-tail 나눌 지점 계산 과정
    peak1 = np.argmax(curvatures)
    curvatures_temp = curvatures.copy()
    curvatures_temp[peak1] = -1
    peak2 = np.argmax(curvatures_temp)

    #스켈레톤화 된 경로 계산
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    start = coords[i]
    sorted_idx = np.argsort(np.linalg.norm(coords - start, axis=1))
    sorted_coords = coords[sorted_idx]

    idx_hb = np.where(sorted_idx == peak1)[0][0]
    idx_bt = np.where(sorted_idx == peak2)[0][0]
    if idx_hb > idx_bt:
        idx_hb, idx_bt = idx_bt, idx_hb

    head_pts = sorted_coords[:idx_hb]
    body_pts = sorted_coords[idx_hb:idx_bt]
    tail_pts = sorted_coords[idx_bt:]

    #부위 별 마스크 형성
    shape = total_mask.shape
    head_mask_t = np.zeros(shape, dtype=np.uint8)
    body_mask_t = np.zeros(shape, dtype=np.uint8)
    head_mask_c = np.zeros(shape, dtype=np.uint8)
    body_mask_c = np.zeros(shape, dtype=np.uint8)

    for z, y, x in head_pts:
        head_mask_t[z, y, x] = total_mask[z, y, x]
        head_mask_c[z, y, x] = com_mask[z, y, x]
    for z, y, x in body_pts:
        body_mask_t[z, y, x] = total_mask[z, y, x]
        body_mask_c[z, y, x] = com_mask[z, y, x]

    #전체 마스크-(머리+몸통)=꼬리
    head_body_mask_t = head_mask_t + body_mask_t
    head_body_mask_c = head_mask_c + body_mask_c
    tail_mask_t = total_mask.copy()
    tail_mask_t[head_body_mask_t > 0] = 0
    tail_mask_t = extract_main_component(tail_mask_t)
    tail_mask_c = com_mask.copy()
    tail_mask_c[head_body_mask_c > 0] = 0
    tail_mask_c = extract_main_component(tail_mask_c)

    return head_mask_t, body_mask_t, tail_mask_t, head_mask_c, body_mask_c, tail_mask_c
