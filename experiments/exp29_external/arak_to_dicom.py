"""Arak (Kaggle): PNG из ПО денситометра, кость тёмная на светлом фоне (так показывает аппарат). В DICOM такой вид — MONOCHROME1;
тег при выгрузке в PNG потерян, поэтому ставим его, чтобы конвейер прочитал снимок так, как прочитал бы исходный DICOM."""
import os, sys
import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True); n = 0
for f in sorted(os.listdir(src)):
    if not f.lower().endswith(".png"):
        continue
    a = np.asarray(Image.open(os.path.join(src, f)).convert("L"), dtype=np.uint8)
    meta = FileMetaDataset(); meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage; meta.MediaStorageSOPInstanceUID = generate_uid(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid(); ds.SeriesInstanceUID = generate_uid(); ds.Modality = "OT"
    ds.Rows, ds.Columns = a.shape; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0; ds.PixelData = a.tobytes()
    ds.save_as(os.path.join(dst, os.path.splitext(f)[0] + ".dcm"), write_like_original=False); n += 1
print("сконвертировано:", n)
