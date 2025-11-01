import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from skimage.restoration import denoise_bilateral
import imutils

IN = "distorted_qr.png"
OUT_DIR = "restoration_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def try_decode(img):
    qr = cv2.QRCodeDetector()
    data, points, _ = qr.detectAndDecode(img)
    if data:
        print("✅ QR decoded:", data)
        return data
    else:
        print("❌ QR not detected.")
        return None

def save(img, name):
    path = os.path.join(OUT_DIR, name)
    cv2.imwrite(path, img)
    print("Saved:", path)

def try_decode_cv2(img):
    qrDecoder = cv2.QRCodeDetector()
    data, points, straight_qrcode = qrDecoder.detectAndDecode(img)
    if isinstance(data, str) and len(data) > 0:
        return data
    return None

def try_decode_pyzbar(img):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    decoded = try_decode(img)
    if decoded:
        return [d.data.decode("utf-8", errors="ignore") for d in decoded]
    return None

def enhance_image_pil(img_bgr):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pil = pil.filter(ImageFilter.MedianFilter(size=3))
    pil = ImageEnhance.Sharpness(pil).enhance(1.5)
    pil = ImageEnhance.Contrast(pil).enhance(1.3)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def adaptive_threshold_and_morph(img_gray):
    den = denoise_bilateral(img_gray, sigma_color=0.05, sigma_spatial=15, multichannel=False)
    den = (den * 255).astype(np.uint8)
    th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 25, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mor = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mor

def find_qr_like_contour(binary):
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            candidates.append((area, approx))
    candidates.sort(key=lambda x: -x[0])
    if candidates:
        return candidates[0][1].reshape(4,2)
    return None

def four_point_transform(image, pts):
    rect = imutils.perspective.order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warp

def upscale(img, scale=2):
    up = img.copy()
    for _ in range(int(np.log2(scale)) if scale>1 else 0):
        up = cv2.pyrUp(up)
    return up

def inpaint_using_mask(img_gray, mask):
    color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    inpainted = cv2.inpaint(color, mask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)

def pipeline_attempts(img_bgr):
    results = []

    e1 = enhance_image_pil(img_bgr)
    save(e1, "enhanced_step1.png")
    res = try_decode_cv2(e1)
    if res:
        results.append(("cv2_enhance", res))
    res2 = try_decode_pyzbar(e1)
    if res2:
        results.append(("pyzbar_enhance", res2))

    gray = cv2.cvtColor(e1, cv2.COLOR_BGR2GRAY)
    save(gray, "gray_after_enhance.png")
    thr = adaptive_threshold_and_morph(gray)
    save(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR), "adaptive_threshold.png")

    res = try_decode_cv2(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
    if res:
        results.append(("cv2_thresh", res))
    res2 = try_decode_pyzbar(cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR))
    if res2:
        results.append(("pyzbar_thresh", res2))

    quad = find_qr_like_contour(thr)
    if quad is not None:
        warped = four_point_transform(img_bgr, quad)
        save(warped, "warped_from_quad.png")
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        t2 = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 21, 6)
        save(cv2.cvtColor(t2, cv2.COLOR_GRAY2BGR), "warped_threshold.png")
        res = try_decode_cv2(warped)
        if res:
            results.append(("cv2_warp", res))
        res2 = try_decode_pyzbar(warped)
        if res2:
            results.append(("pyzbar_warp", res2))

        up = upscale(warped, scale=2)
        save(up, "warped_upscaled.png")
        res = try_decode_cv2(up)
        if res:
            results.append(("cv2_warp_up", res))
        res2 = try_decode_pyzbar(up)
        if res2:
            results.append(("pyzbar_warp_up", res2))

    inv = cv2.bitwise_not(thr)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(inv, connectivity=8)
    mask = np.zeros_like(inv)
    for i in range(1, nb_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 500:
            mask[output == i] = 255
    save(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "inpaint_mask.png")
    if mask.sum() > 0:
        inpainted = inpaint_using_mask(gray, mask)
        save(cv2.cvtColor(inpainted, cv2.COLOR_GRAY2BGR), "inpainted.png")
        res = try_decode_cv2(cv2.cvtColor(inpainted, cv2.COLOR_GRAY2BGR))
        if res:
            results.append(("cv2_inpaint", res))
        res2 = try_decode_pyzbar(cv2.cvtColor(inpainted, cv2.COLOR_GRAY2BGR))
        if res2:
            results.append(("pyzbar_inpaint", res2))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    save(cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR), "opened.png")
    res = try_decode_cv2(cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR))
    if res:
        results.append(("cv2_open", res))
    res2 = try_decode_pyzbar(cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR))
    if res2:
        results.append(("pyzbar_open", res2))

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        angles = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            angle = np.degrees(np.arctan2((y2-y1),(x2-x1)))
            angles.append(angle)
        median_angle = np.median(angles)
        deskew = imutils.rotate_bound(img_bgr, -median_angle)
        save(deskew, "deskewed.png")
        res = try_decode_cv2(deskew)
        if res:
            results.append(("cv2_deskew", res))
        res2 = try_decode_pyzbar(deskew)
        if res2:
            results.append(("pyzbar_deskew", res2))

    return results

def main():
    if not os.path.exists(IN):
        print(f"Input file '{IN}' not found. Pune distored_qr.png în directorul curent.")
        return
    img = cv2.imread(IN)
    if img is None:
        print("Nu pot citi imaginea (cv2.imread returned None). Verifică formatul.")
        return

    print("Imagine încărcată:", IN)
    save(img, "original.png")

    results = pipeline_attempts(img)

    found = {}
    for tag, res in results:
        if isinstance(res, list):
            for item in res:
                found[item] = found.get(item, []) + [tag]
        else:
            found[res] = found.get(res, []) + [tag]

    if found:
        print("\n--- Decodări găsite ---")
        for text, methods in found.items():
            print("Decoded text:", text)
            print("Found by:", methods)
            print()
    else:
        print("\nN-am reușit să decodăm automat. Ce poți încerca în continuare:")
        print(" - Ajustează parametrii din funcțiile adaptive_threshold_and_morph și inpaint (block size, C, area threshold).")
        print(" - Deschide imaginile salvate în folderul 'restoration_outputs' și inspectează ce pas arată cel mai promițător.")
        print(" - Dacă ai acces la versiuni multiple ale imaginii (scansări diferite), rulează scriptul pe fiecare.")
        print(" - Poți încerca upscale mai agresiv (scale=4) sau metode de super-resolution (cv2.dnn_superres).")
        print(" - Dacă vrei, încarcă aici una dintre imaginile intermediare și încerc eu următorii pași.")

if __name__ == "__main__":
    main()
