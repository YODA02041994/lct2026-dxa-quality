import os, sys, collections
import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, SecondaryCaptureImageStorage
src, dst = sys.argv[1], sys.argv[2]
clean = len(sys.argv) > 3 and sys.argv[3] == "--clean"
import cv2
sizes, modes, n = collections.Counter(), collections.Counter(), 0
for cls in ("normal", "osteophenia", "osteoporosis"):
    d = os.path.join(src, cls); os.makedirs(os.path.join(dst, cls), exist_ok=True)
    for f in sorted(os.listdir(d)):
        im = Image.open(os.path.join(d, f)); modes[im.mode] += 1; sizes[im.size] += 1
        a = np.asarray(im.convert("L"), dtype=np.uint8)
        if clean:                                  # убрать впечатанную цветную разметку денситометра (линии ROI, подписи L1–L4)
            rgb = np.asarray(im.convert("RGB")).astype(np.int16)
            mask = ((rgb.max(2) - rgb.min(2)) > 30).astype(np.uint8)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            a = cv2.inpaint(a, mask * 255, 3, cv2.INPAINT_TELEA)
        meta = FileMetaDataset(); meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage; meta.MediaStorageSOPInstanceUID = generate_uid(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = SecondaryCaptureImageStorage; ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = generate_uid(); ds.SeriesInstanceUID = generate_uid(); ds.Modality = "OT"
        ds.Rows, ds.Columns = a.shape; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0; ds.PixelData = a.tobytes()
        ds.save_as(os.path.join(dst, cls, os.path.splitext(f)[0].replace(" ", "_").replace("(", "").replace(")", "") + ".dcm"), write_like_original=False)
        n += 1
print("сконвертировано:", n, "| режимы:", dict(modes), "| размеры (топ-6):", sizes.most_common(6))
