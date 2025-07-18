import numpy as np
from sklearn.cluster import KMeans
from rt_utils import RTStructBuilder
import os
from stl import mesh as stl_mesh
from skimage import measure
import SimpleITK as sitk

def extract_roi_mask(dicom_path, rtstruct_path, roi_name="Pancreas"):
    rtstruct = RTStructBuilder.create_from(dicom_path, rtstruct_path)
    mask = rtstruct.get_roi_mask_by_name(roi_name)
    return mask

def kmeans_split(mask, spacing, reorder=True):
    coords = np.array(np.nonzero(mask)).T
    if coords.shape[0] < 3:
        raise ValueError("Too few points for clustering")

    kmeans = KMeans(n_clusters=3, random_state=0).fit(coords)
    labels = kmeans.labels_

    masks = [np.zeros_like(mask, dtype=np.uint8) for _ in range(3)]
    for i, (z, y, x) in enumerate(coords):
        masks[labels[i]][z, y, x] = 1

    if reorder:
        coms = [np.array(np.mean(np.array(np.nonzero(m)).T, axis=0)) for m in masks]
        head_idx = np.argmin([c[0] for c in coms]) 
        tail_idx = np.argmax([c[1] for c in coms])  
        body_idx = list({0, 1, 2} - {head_idx, tail_idx})[0]
        ordered_masks = [masks[head_idx], masks[body_idx], masks[tail_idx]]
    else:
        ordered_masks = masks
    return ordered_masks

def save_mask_as_stl(mask, output_path, spacing):
    verts, faces, _, _ = measure.marching_cubes(mask, level=0, spacing=spacing)
    pancreas_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            pancreas_mesh.vectors[i][j] = verts[f[j], :]
    pancreas_mesh.save(output_path)

def dicom_spacing(dicom_path):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_files)
    image = reader.Execute()
    spacing = image.GetSpacing() 
    return spacing

def get_slice_range(mask, axis=0):
    projection = mask.sum(axis=(1, 2)) if axis == 0 else mask.sum(axis=(0, 2)) if axis == 1 else mask.sum(axis=(0, 1))
    nonzero = np.nonzero(projection)[0]
    if len(nonzero) == 0:
        return -1, -1
    return nonzero[0], nonzero[-1]

def main():
    print('K-means 기반 Head/Body/Tail 분할 및 STL 저장')
    rt = input('RTStruct 파일 경로를 입력하세요: ').strip('"')
    dicom = input('DICOM 시리즈 폴더 경로를 입력하세요: ').strip('"')
    out_dir = input('STL 파일을 저장할 폴더 경로를 입력하세요: ').strip('"')

    os.makedirs(out_dir, exist_ok=True)

    mask_total = extract_roi_mask(dicom, rt)

    print("K-means 기반 분할 중...")
    head, body, tail = kmeans_split(mask_total, spacing=dicom_spacing(dicom))

    h_start, h_end = get_slice_range(head, axis=0)
    b_start, b_end = get_slice_range(body, axis=0)
    t_start, t_end = get_slice_range(tail, axis=0)

    print(f"Head 마스크 슬라이스 범위: {h_start} ~ {h_end}")
    print(f"Body 마스크 슬라이스 범위: {b_start} ~ {b_end}")
    print(f"Tail 마스크 슬라이스 범위: {t_start} ~ {t_end}")

    save_mask_as_stl(head, os.path.join(out_dir, "pancreas_head.stl"), spacing=dicom_spacing(dicom))
    save_mask_as_stl(body, os.path.join(out_dir, "pancreas_body.stl"), spacing=dicom_spacing(dicom))
    save_mask_as_stl(tail, os.path.join(out_dir, "pancreas_tail.stl"), spacing=dicom_spacing(dicom))

    print("STL 저장 완료!")

if __name__ == "__main__":
    main()
