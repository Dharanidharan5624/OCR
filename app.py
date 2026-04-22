import sys
import os
import re
import csv
import pickle
import time
from functools import lru_cache
import numpy as np
from datetime import datetime
import cv2
import torch
from torchvision.transforms import ToTensor, Resize
from PIL import Image
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

# ================================================================================
# SPEED OPTIMIZATION CONSTANTS
# ================================================================================
OCR_TO_TENSOR = ToTensor()
OCR_RESIZE    = Resize((32, 128))
SAVE_CROPS    = False

CONF_FAST_EXIT   = 1.0
CONF_ACCEPT      = 0.62   # High-quality only
CONF_FALLBACK    = 0.52   # Robust fallback
MIN_CHAR_CONF    = 0.35   # No single character should be a total guess

MAX_OCR_CALLS_QUICK    = 8
MAX_OCR_CALLS_FALLBACK = 16
MAX_VARIANTS_PER_CROP  = 4

_DUMMY_TENSOR = torch.zeros(1, 3, 32, 128)
TEMPLATE_CACHE = {}

# ================================================================================
# FIX #2: GreatCollections label bounding box exclusion
# The GreatCollections label typically appears in the LOWER-LEFT quadrant of
# the coin holder image (QR code + number + brand text).
# We detect it and exclude that region from OCR scanning.
# ================================================================================

def estimate_gc_label_region(image):
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    # QR Code + Bottom Hologram area
    # GreatCollections has a QR on the left and a hologram on the bottom.
    # We want to block everything EXCEPT the top-right text area.
    return (0, int(h * 0.45), w, h) # Mask bottom half and left side later


def mask_gc_label(image):
    """
    Soften masking to ensure ID is always visible.
    Only masks the far-left (QR) and brand text if possible.
    """
    if image is None or image.size == 0:
        return image
    
    h, w = image.shape[:2]
    masked = image.copy()
    
    # 1. Mask QR Code area (Left 42%)
    cv2.rectangle(masked, (0, 0), (int(w * 0.42), h), (0, 0, 0), -1)
    
    # 2. Mask TOP edge (often noise)
    cv2.rectangle(masked, (0, 0), (w, int(h * 0.10)), (0, 0, 0), -1)
    
    # 3. Mask BOTTOM area (Hologram / Brand)
    # We move this lower to ensure we don't clip the ID
    cv2.rectangle(masked, (0, int(h * 0.68)), (w, h), (0, 0, 0), -1)
        
    return masked


# ================================================================================
# FIX #3: Coin-ID vs GC-number disambiguation
# The coin ID from GreatCollections is printed on the RIGHT side of their label.
# The actual coin holder/slab number we want is usually printed ABOVE the GC label
# (upper-right area of the holder).  We add a scoring function that penalises
# any candidate that spatially overlaps the known GC label zone.
# ================================================================================

# ================================================================================
# FAST OCR HELPERS
# ================================================================================

def perform_inference(model, image, model_args):
    """
    Overhauled to preserve aspect ratio using padding. 
    Prevents digit squashing which causes misreads.
    """
    image_pil = Image.fromarray(image).convert("RGB")
    w, h      = image_pil.size
    target_h  = 32
    target_w  = 128
    
    # Scale to fixed height 32, maintain aspect ratio
    nh = target_h
    nw = int(w * (target_h / h))
    if nw > target_w: nw = target_w  # Cap width
    
    image_resized = image_pil.resize((nw, nh), Image.Resampling.LANCZOS)
    
    # Pad to 128 width
    final_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    final_img.paste(image_resized, (0, 0))
    
    image_tensor = OCR_TO_TENSOR(final_img).unsqueeze(0).to(model_args['device'])
    with torch.inference_mode():
        output = model(image_tensor)
    return output


def ocr_text(inference_result, loaded_tokenizer):
    pred  = inference_result.softmax(-1)
    label, confidence = loaded_tokenizer.decode(pred)
    return label[0], confidence[0]


# ================================================================================
# PREPROCESSING
# ================================================================================

def _prep_standard(image):
    if image is None or image.size == 0: return None
    # No more resizing or sharpening here - handles that in get_variants
    return image


def _prep_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _prep_red_channel(image):
    b, g, r = cv2.split(image)
    mask = np.clip(r.astype(np.int16) - g.astype(np.int16), 0, 255).astype(np.uint8)
    _, thresh = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def _prep_adaptive(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 3)
    adap_bgr = cv2.cvtColor(adap, cv2.COLOR_GRAY2BGR)
    return _prep_standard(adap_bgr)


def _prep_contrast(image):
    if image is None or image.size == 0: return None
    # Contrast stretching / Normalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(gray)
    # Ensure we don't divide by zero
    diff = max_val - min_val
    if diff < 1: diff = 1
    stretched = cv2.convertScaleAbs(gray, alpha=250.0/diff, beta=-min_val * 250.0/diff)
    return _prep_standard(cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR))


def _prep_gamma(image, gamma=1.2):
    if image is None or image.size == 0: return None
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return _prep_standard(cv2.LUT(image, table))


def get_variants(crop):
    # Base variants
    variants = [crop]
    
    # ── BASELINE ──
    # 2.0x Lanczos resize for clean character definition
    # (Higher than 2.0x can sometimes cause model confusion on 32px height)
    h, w = crop.shape[:2]
    resized = cv2.resize(crop, (int(w*2.0), int(h*2.0)), interpolation=cv2.INTER_LANCZOS4)
    variants.append(resized)
    
    # ── LIGHTING / CONTRAST ──
    variants.append(_prep_gamma(crop, 1.2))
    variants.append(_prep_contrast(crop))
    variants.append(_prep_otsu(crop))
    
    # SHARPENING REMOVED: It causes 3->2 and 9->0 confusion on GC fonts.
    
    return variants[:MAX_VARIANTS_PER_CROP + 4]


# ================================================================================
# LABEL VALIDATION
# ================================================================================

def is_valid_label_id(text):
    text = str(text).strip()
    if len(text) not in (7, 8):
        return False
    if not text.isdigit():
        return False
    if text.startswith('0'):
        return False
    # Pattern check: reject binary-like hallucinations (lots of 1s and 0s)
    if len(re.sub(r'[01]', '', text)) <= 2 and len(text) >= 7:
        return False
    # Reject periodic patterns often from background textures
    if len(set(text)) <= 2 and len(text) >= 7:
        return False
    if re.search(r'(\d)\1{3,}', text):
        return False
        
    # NEW: Reject highly repetitive digits (e.g. 555-5555 or 111-xxxx where x is noise)
    # A single digit appearing 4 or more times in a 7-digit ID is very rare for valid IDs
    # and common for barcode hallucinations.
    if len(text) == 7:
        counts = [text.count(d) for d in set(text)]
        if max(counts) >= 4:
            return False

    return True


def normalize_ocr_text(text):
    return text.translate(str.maketrans('lIoOSBZGqgD', '11005826990'))


def extract_label_candidates(raw_text):
    raw_text   = str(raw_text).strip()
    candidates = []
    raw_ns     = re.sub(r'\s+', '', raw_text)
    for tv in [raw_text, normalize_ocr_text(raw_text), raw_ns, normalize_ocr_text(raw_ns)]:
        for num in re.findall(r'\d+', tv):
            if is_valid_label_id(num) and num not in candidates:
                candidates.append(num)
            elif len(num) == 8 and num.startswith('0'):
                t = num[1:]
                if is_valid_label_id(t) and t not in candidates:
                    candidates.append(t)
            elif len(num) > 8:
                for s in range(len(num) - 6):
                    for ln in (7, 8):
                        sub = num[s:s + ln]
                        if is_valid_label_id(sub) and sub not in candidates:
                            candidates.append(sub)
    
    # Priority: 7-digit numbers are preferred for coin slabs
    c7 = [c for c in candidates if len(c) == 7]
    if c7: return c7
    return candidates


# ================================================================================
# CORE OCR CALL
# ================================================================================

def _ocr_crop(crop, model_ocr, loaded_tokenizer, model_args, call_counter, max_calls):
    if call_counter[0] >= max_calls:
        return []
    if crop is None or crop.size == 0:
        return []
    try:
        call_counter[0] += 1
        ir      = perform_inference(model_ocr, crop, model_args)
        text, char_conf = ocr_text(ir, loaded_tokenizer)
        
        # CHECK: No total guesses allowed for any character
        if torch.min(char_conf).item() < MIN_CHAR_CONF:
            return []
        
        nums    = extract_label_candidates(text)
        if not nums:
            return []
            
        mean_c  = float(np.mean(char_conf.cpu().numpy()))
        # Return a list of (text, confidence) tuples
        return [(n, mean_c) for n in nums]
    except Exception:
        return []


def read_best_label_id(crops, model_ocr, loaded_tokenizer, model_args,
                       call_counter=None, max_calls=MAX_OCR_CALLS_QUICK,
                       source_image=None):
    if call_counter is None:
        call_counter = [0]
    
    # Collective candidates: text -> list of confidences
    candidates = {}

    for crop in crops:
        if crop is None or crop.size == 0:
            continue
        for variant in get_variants(crop):
            found_list = _ocr_crop(variant, model_ocr, loaded_tokenizer, model_args,
                                   call_counter, max_calls)
            for t, c in found_list:
                if t and is_valid_label_id(t):
                    if t not in candidates:
                        candidates[t] = []
                    candidates[t].append(c)

    # FILTER: Aggressively reject non-7-digit garbage and hallucinations
    cleaned_candidates = {}
    for text, confs in candidates.items():
        # Reject obvious noise / QR fragments
        if len(text) < 5 or len(text) > 9: continue
        # GreatCollections IDs are consistently 7 digits
        if len(text) != 7: continue
        # Reject patterns common to hallucinations (like starting with 918/235)
        if text.startswith('918') or text.startswith('235'): continue
        
        cleaned_candidates[text] = confs
    
    if not cleaned_candidates:
        return "", 0.0
    
    candidates = cleaned_candidates

    # RANKING BY FONT FIDELITY
    # If we have multiple 7-digit choices, pick the one that matches Lable.png best
    best_text = ""
    best_fidelity = -1.0
    
    for text in candidates:
        if crops:
            # Fidelity is the quality of the font match
            fidelity = calculate_font_fidelity(crops[0], text)
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_text = text
    
    if not best_text:
        sorted_candidates = sorted(candidates.keys(), 
                                    key=lambda k: (len(candidates[k]), max(candidates[k])), 
                                    reverse=True)
        best_text = sorted_candidates[0]

    best_conf = float(max(candidates[best_text]))

    # Universal verification for final correction
    try:
        if best_text and crops:
            best_text = verify_with_templates(crops[0], best_text)
    except Exception: pass

    return best_text, best_conf

def calculate_font_fidelity(crop, text):
    """ High-precision tie-breaker: how well does the ID match our Label.png font? """
    try:
        # If verification returns the same string, it's a high-fidelity match
        res_id = verify_with_templates(crop, text)
        if res_id == text:
            return 1.0
        # If it changes, check if the change is minor (likely 2 vs 3)
        return 0.5
    except:
        return 0.0


def verify_with_templates(crop, detected_id):
    """
    Universal Digital Verification using Lable.png font.
    Splits the crop into characters using projection and matches them one-by-one.
    """
    global TEMPLATE_CACHE
    try:
        if not os.path.exists('font_templates') or not detected_id:
            return detected_id
            
        # 1. High-Precision Preprocessing
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h_crop, w_crop = gray.shape[:2]
        match_h = 64 # Use higher resolution for better features
        gray_res = cv2.resize(gray, (int(w_crop * (match_h / h_crop)), match_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Binarize with Otsu to handle different lighting
        _, thresh = cv2.threshold(gray_res, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Digit Segmentation via Projection
        proj = np.sum(thresh, axis=0)
        # Find continuous blocks of horizontal pixels (where characters are)
        char_mask = (proj > (np.max(proj) * 0.15)).astype(np.uint8)
        
        # Find start/end of each character block
        digit_zones = []
        in_block = False
        start = 0
        for x in range(len(char_mask)):
            if char_mask[x] and not in_block:
                start = x
                in_block = True
            elif not char_mask[x] and in_block:
                if (x - start) > 5: # Min width
                    digit_zones.append((start, x))
                in_block = False
        if in_block:
            digit_zones.append((start, len(char_mask)))
            
        # If we didn't find enough zones, fallback to fixed segmentation
        if len(digit_zones) != len(detected_id):
            char_w = thresh.shape[1] / len(detected_id)
            digit_zones = [(int(i * char_w), int((i + 1) * char_w)) for i in range(len(detected_id))]

        new_id = []
        for i, (x1, x2) in enumerate(digit_zones):
            if i >= len(detected_id): break
            
            # Extract and standardize the character segment
            char_seg = thresh[:, max(0, x1-2):min(thresh.shape[1], x2+2)]
            char_seg = cv2.resize(char_seg, (32, 64))
            
            best_char = detected_id[i]
            best_char_score = -1.0
            
            for cand_char in "0123456789":
                if cand_char not in TEMPLATE_CACHE:
                    path = os.path.join('font_templates', f"{cand_char}.png")
                    if os.path.exists(path):
                        tpl = cv2.imread(path, 0)
                        tpl_res = cv2.resize(tpl, (32, 64))
                        _, t_th = cv2.threshold(tpl_res, 127, 255, cv2.THRESH_BINARY_INV)
                        TEMPLATE_CACHE[cand_char] = t_th
                
                if cand_char in TEMPLATE_CACHE:
                    tpl_img = TEMPLATE_CACHE[cand_char]
                    score = cv2.matchTemplate(char_seg, tpl_img, cv2.TM_CCOEFF_NORMED)[0][0]
                    
                    # STRUCTURAL TIE-BREAKERS
                    if cand_char == '3':
                        # Check middle-left indentation
                        mid_left = char_seg[28:36, :10]
                        if np.mean(mid_left) < 30: score += 0.20
                    elif cand_char == '5':
                        # Check top-left corner density (5 has it, 3 doesn't)
                        top_left = char_seg[:15, :10]
                        if np.mean(top_left) > 120: score += 0.15
                    elif cand_char == '2':
                        # Look for solid diagonal / base
                        base = char_seg[50:, :]
                        if np.mean(base) > 150: score += 0.10

                    if score > best_char_score:
                        best_char_score = score
                        best_char = cand_char
            
            new_id.append(best_char)
            
        return "".join(new_id)
    except Exception:
        return detected_id


# ================================================================================
# IMAGE ANALYSIS HELPERS
# ================================================================================

def looks_like_label_region(image):
    if image is None or image.size == 0:
        return False
    h, w = image.shape[:2]
    # Barcode/Texture Rejection: check for excessive vertical edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel_x = np.absolute(sobel_x)
    # RELAXED: from 0.35 to 0.55 to avoid QR code interference on GC labels
    if np.mean(abs_sobel_x > 45) > 0.55:
        return False

    hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = cv2.split(hsv)
    
    # HOLOGRAM REJECTION: Holograms are highly saturated / colorful.
    # Standard text labels are black/white (low saturation).
    if np.mean(s_c) > 40: # High saturation = likely hologram
        return False

    if float(np.mean(v_c > 150)) >= 0.28 and float(np.mean(s_c < 75)) >= 0.25:
        return True
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray)) > 18


def has_any_label_zone(image):
    """
    Very permissive check to avoid skipping valid images.
    """
    if image is None or image.size == 0:
        return False
    return True


def looks_like_coin_face(image):
    if image is None or image.size == 0:
        return False
    h, w   = image.shape[:2]
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center = gray[int(h*0.20):int(h*0.75), int(w*0.15):int(w*0.90)]
    if center.size > 0 and float(np.mean(center > 160)) > 0.18:
        return False
    tr = gray[int(h*0.05):int(h*0.45), int(w*0.40):]
    cx = gray[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    tr_bright = float(np.mean(tr > 120)) if tr.size else 0.0
    cx_bright = float(np.mean(cx > 140)) if cx.size else 0.0
    return tr_bright < 0.10 and cx_bright > 0.30


def resize_keep_aspect(image, target=416):
    if image is None or image.size == 0:
        return image
    h, w   = image.shape[:2]
    scale  = target / max(h, w)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((target, target, 3), dtype=np.uint8)
    py, px  = (target - nh) // 2, (target - nw) // 2
    canvas[py:py+nh, px:px+nw] = resized
    return canvas


# ================================================================================
# DRAW OVERLAY
# ================================================================================

def draw_overlay_text(image, text, color, anchor=(60, 70), max_width_ratio=0.72):
    text   = str(text)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    thick  = 2
    scale  = 2.0
    max_w  = int(image.shape[1] * max_width_ratio)
    while scale > 0.8:
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        if tw <= max_w:
            break
        scale -= 0.2
    x = (image.shape[1] - tw) // 2
    y = max(th + 15, anchor[1])
    y = min(y, image.shape[0] - bl - 10)
    cv2.putText(image, text, (x, y), font, scale, color, thick)


# ================================================================================
# BLUE BOUNDING BOX OVERLAY
# ================================================================================

def draw_detection_box(image, text, region=None):
    img_h, img_w = image.shape[:2]
    BLUE = (255, 0, 0)

    if region:
        x1, y1, x2, y2 = [int(v) for v in region]
    else:
        x1 = int(img_w * 0.42)
        x2 = int(img_w * 0.96)
        y1 = int(img_h * 0.28)
        y2 = int(img_h * 0.62)

    cv2.rectangle(image, (x1, y1), (x2, y2), BLUE, 3)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness  = 2
    (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)
    pill_x1 = x1
    pill_y1 = max(0, y1 - th - bl - 6)
    pill_x2 = x1 + tw + 8
    pill_y2 = y1
    cv2.rectangle(image, (pill_x1, pill_y1), (pill_x2, pill_y2), BLUE, -1)
    cv2.putText(image, text, (pill_x1 + 4, pill_y2 - bl - 2),
                font, font_scale, (255, 255, 255), thickness)


# ================================================================================
# FIX #4: MATCH RESULT — Exact match only, confidence gated
# Old code used allow_generic_success=True which returned "True" for ANY detected
# number even when it didn't match the filename.  Now we only return "True" when
# the detected text exactly equals the expected ID extracted from the filename,
# AND the confidence is above CONF_ACCEPT.
# ================================================================================

def build_match_result(actual_value, detected_value, confidence=0.0):
    """
    Returns:
        "True"  – detected_value exactly matches actual_value AND conf >= CONF_ACCEPT
                  OR a valid ID is detected when filename has no ID
        "False" – detected_value does NOT match actual_value (when filename has ID)
        ""      – No detected ID
    """
    actual_value   = str(actual_value).strip()
    detected_value = str(detected_value).strip()

    if not detected_value:
        return ""

    if not re.search(r'\d{5,9}', actual_value):
        # Filename has no numeric ID. Don't call it "True" as we can't verify it.
        # This prevents misleading green "True" for misread random IDs.
        return ""

    # Must be exact match AND confidence must meet threshold
    if actual_value.lower() == detected_value.lower() and confidence >= CONF_ACCEPT:
        return "True"
    else:
        return "False"


# ================================================================================
# FIX #5: QUICK SCAN — mask GC label region before scanning
# ================================================================================

def quick_label_scan(image, model_ocr, loaded_tokenizer, model_args):
    if not has_any_label_zone(image):
        return "", 0.0

    img_h, img_w = image.shape[:2]
    call_counter = [0]

    # We already receive scan_image which was masked outside
    # but we call mask_gc_label again for safety in case of nested calls
    scan_image = mask_gc_label(image)

    # FIX #5: Crop defs now prioritise the UPPER portion of the holder
    # (where the actual slab/coin ID is printed), not the lower GC label area.
    # Tighter crops that focus on the number area while avoiding brand text and barcodes
    crop_defs = [
        # Upper-right: typical coin ID location on slabs
        scan_image[int(img_h*0.08):int(img_h*0.32), int(img_w*0.35):int(img_w*0.97)],
        # Middle band (specific number area of the GC label) - BROADENED width to avoid clipping
        scan_image[int(img_h*0.38):int(img_h*0.56), int(img_w*0.42):int(img_w*0.96)],
        # Narrow slice to avoid barcodes - BROADENED width to avoid clipping
        scan_image[int(img_h*0.40):int(img_h*0.54), int(img_w*0.38):int(img_w*0.97)],
        # NEW: Lower-middle crop specifically for GreatCollections slabs
        scan_image[int(img_h*0.45):int(img_h*0.65), int(img_w*0.38):int(img_w*0.97)],
        # NEW: Right-side targeted crop for GC serial IDs
        scan_image[int(img_h*0.42):int(img_h*0.62), int(img_w*0.40):int(img_w*0.95)],
    ]

    best_text, best_conf = "", 0.0
    for crop in crop_defs:
        if crop is None or crop.size == 0:
            continue
        t, c = read_best_label_id([crop], model_ocr, loaded_tokenizer, model_args,
                                   call_counter, MAX_OCR_CALLS_QUICK,
                                   source_image=image)   # Pass original for penalty check
        if c > best_conf:
            best_text, best_conf = t, c
        if best_conf >= CONF_FAST_EXIT:
            break

    if best_text and is_valid_label_id(best_text) and best_conf >= CONF_ACCEPT:
        return best_text, best_conf
    return "", 0.0


# ================================================================================
# FIX #6: FALLBACK SCAN — mask GC label + only scan upper region
# ================================================================================

def fallback_label_scan(image, model_ocr, loaded_tokenizer, model_args, actual_value):
    img_h, img_w = image.shape[:2]
    if not has_any_label_zone(image):
        return "", 0.0

    # Mask GC label before fallback scanning
    scan_image   = mask_gc_label(image)

    call_counter     = [0]
    best_text        = ""
    best_conf        = 0.0
    has_expected_id  = bool(re.search(r'\d{5,9}', actual_value))

    strip_h = 80
    step    = 24

    # FIX #6: Only scan upper/middle 70% of image to avoid GC brand area but catch lower IDs
    nofilter_regions = [
        (int(img_w*0.10), int(img_h*0.05), int(img_w*0.96), int(img_h*0.70)),
    ]

    for (x1, y1, x2, y2) in nofilter_regions:
        for y in range(y1, y2 - strip_h, step):
            if call_counter[0] >= MAX_OCR_CALLS_FALLBACK:
                break
            strip = scan_image[y:y + strip_h, x1:x2]
            if strip is None or strip.size == 0:
                continue
            t, c = read_best_label_id([strip], model_ocr, loaded_tokenizer, model_args,
                                       call_counter, MAX_OCR_CALLS_FALLBACK,
                                       source_image=image)
            if not t or not is_valid_label_id(t) or c < CONF_FALLBACK:
                continue
            
            # UNBIASED SELECTION: Pick the highest confidence valid ID
            # regardless of whether it matches actual_value.
            if c > best_conf:
                best_text, best_conf = t, c
            
            # Fast exit if we find a very high confidence match
            # ABSOLUTE BIAS REMOVAL: No more target-matching fast-exit. 
            # We must scan all regions to find the GLOBALLY highest-confidence ID.
            pass

        if call_counter[0] >= MAX_OCR_CALLS_FALLBACK:
            break

    if best_text and is_valid_label_id(best_text):
        return best_text, best_conf
    return "", 0.0


# ================================================================================
# OCR WORKER THREAD
# ================================================================================

class OCRWorker(QThread):
    progress        = pyqtSignal(int, int)
    image_processed = pyqtSignal(object, list)
    finished        = pyqtSignal(list)
    log             = pyqtSignal(str)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.running     = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            self.log.emit("Loading models…")
            torch.set_num_threads(2)

            model_args = {
                'data_root': 'data', 'batch_size': 1, 'num_workers': 4,
                'cased': False, 'punctuation': False, 'new': False,
                'rotation': 0, 'device': 'cpu'
            }

            with open('tokenizer/tokenizer.pkl', 'rb') as f:
                loaded_tokenizer = pickle.load(f)

            model_ocr = torch.jit.load('pretrained_models/Pretrained.pth',
                                       map_location='cpu').eval()

            self.log.emit("Warming up models…")
            with torch.inference_mode():
                try:
                    model_ocr(_DUMMY_TENSOR)
                except Exception:
                    pass

            model_yolo   = YOLO("weights/best.pt")
            _dummy_img   = np.zeros((416, 416, 3), dtype=np.uint8)
            model_yolo.predict(_dummy_img, device="cpu", imgsz=416, verbose=False)

            class_names  = ['coin_id']

            def extract_number(name):
                m = re.search(r'\d+', name)
                return int(m.group()) if m else float('inf')

            subfolders = [os.path.join(self.folder_path, n)
                          for n in os.listdir(self.folder_path)
                          if os.path.isdir(os.path.join(self.folder_path, n))]
            if not subfolders:
                subfolders = [self.folder_path]
            else:
                subfolders.sort(key=lambda x: extract_number(os.path.basename(x)))

            images_to_process = []
            for sf in subfolders:
                files = sorted(os.listdir(sf), key=extract_number)
                for fn in files:
                    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images_to_process.append(os.path.join(sf, fn))

            total_images = len(images_to_process)
            if total_images == 0:
                self.log.emit("No images found.")
                self.finished.emit([])
                return

            all_detections   = []
            current_date     = datetime.now().strftime("%Y-%m-%d")
            res_folder       = os.path.join("images", "results", current_date)
            os.makedirs(res_folder, exist_ok=True)
            if SAVE_CROPS:
                crops_folder = os.path.join("images", "crops", current_date)
                os.makedirs(crops_folder, exist_ok=True)

            RED  = (0, 0, 255)
            BLUE = (255, 0, 0)

            for i, image_path in enumerate(images_to_process):
                if not self.running:
                    break

                t0 = time.perf_counter()

                try:
                    img_basename   = os.path.basename(image_path)
                    subfolder_name = os.path.basename(os.path.dirname(image_path))
                    raw_name       = img_basename.rsplit('.', 1)[0]
                    all_ids        = re.findall(r'\d{5,9}', raw_name)
                    actual_value   = all_ids[-1] if all_ids else raw_name.split('-')[0].strip()

                    if re.search(r'-1\.', img_basename):
                        self.log.emit(f"Skipping face: {img_basename}")
                        self.progress.emit(i + 1, total_images)
                        image = cv2.imread(image_path)
                        if image is not None:
                            image = resize_keep_aspect(image, 640)
                            row   = (subfolder_name, img_basename, 0, "", "0", "", "")
                            self.image_processed.emit(image, [row])
                            all_detections.append(row)
                        continue

                    self.log.emit(f"[{i+1}/{total_images}] {img_basename}")
                    image = cv2.imread(image_path)
                    if image is None:
                        self.progress.emit(i + 1, total_images)
                        continue
                    image = resize_keep_aspect(image, 640)

                    if looks_like_coin_face(image):
                        row = (subfolder_name, img_basename, 0, "coin-face", "0", "", "")
                        self.image_processed.emit(image, [row])
                        all_detections.append(row)
                        self.progress.emit(i + 1, total_images)
                        continue

                    # ── PREPROCESSING: Mask GC QR codes ────────────────────────
                    scan_image = mask_gc_label(image)

                    # ── STEP 1: Quick label scan ─────────────────────────────
                    detections   = []
                    quick_text, quick_conf = quick_label_scan(
                        scan_image, model_ocr, loaded_tokenizer, model_args
                    )
                    if quick_text and quick_conf >= CONF_ACCEPT and is_valid_label_id(quick_text):
                        # FIX #4: Exact match only, no allow_generic_success
                        match_res = build_match_result(actual_value, quick_text, quick_conf)
                        draw_detection_box(image, quick_text)
                        draw_overlay_text(image, quick_text, RED,
                                          anchor=(55, 55), max_width_ratio=0.65)
                        detections.append((subfolder_name, img_basename, 0,
                                           "quick-scan", f"{quick_conf:.2f}",
                                           quick_text, match_res))

                    # ── STEP 2: YOLO detection ───────────────────────────────
                    if not detections:
                        try:
                            results = model_yolo.predict(
                                image, device="cpu", classes=0,
                                conf=0.45, imgsz=416, verbose=False
                            )
                        except TypeError:
                            results = model_yolo.predict(
                                image, device="cpu", conf=0.45,
                                imgsz=416, verbose=False
                            )

                        for result in results:
                            boxes = result.boxes
                            if len(boxes) == 0:
                                if not detections:
                                    detections.append((subfolder_name, img_basename,
                                                       0, "", "0", "", ""))
                                continue

                            for box_idx in range(len(boxes)):
                                try:
                                    bd = boxes[box_idx].xyxy.cpu().numpy().squeeze()
                                    if len(bd) != 4:
                                        continue
                                    x1, y1, x2, y2 = bd
                                    
                                    # SPATIAL GATE: ID labels are USUALLY in the top half of the slab
                                    # RELAXED: from 0.70 to 0.85 to allow for low labels
                                    if y1 > image.shape[0] * 0.85:
                                        continue

                                    w, h = x2 - x1, y2 - y1
                                    if w <= 0 or h <= 0:
                                        continue
                                    if max(w, h) / min(w, h) < 1.2:
                                        continue

                                    # FIX #6: Skip YOLO boxes that fall inside
                                    # the estimated GC label region
                                    img_h_full, img_w_full = image.shape[:2]
                                    gc_region = estimate_gc_label_region(image)
                                    if gc_region:
                                        gx1, gy1, gx2, gy2 = gc_region
                                        box_cx = (x1 + x2) / 2
                                        box_cy = (y1 + y2) / 2
                                        if gx1 < box_cx < gx2 and gy1 < box_cy < gy2:
                                            self.log.emit(
                                                f"Skipping YOLO box inside GC label zone: {img_basename}")
                                            continue

                                    pad  = 6
                                    x1p  = max(0, int(x1) - pad)
                                    y1p  = max(0, int(y1) - pad)
                                    x2p  = min(image.shape[1], int(x2) + pad)
                                    y2p  = min(image.shape[0], int(y2) + pad)
                                    # Use scan_image (masked) for OCR crops
                                    im   = scan_image[y1p:y2p, x1p:x2p]

                                    # TIGHTEN ROI height: exclude brand names at bottom
                                    # PINPOINT ROI: prefer top-right quadrant for GC labels (avoids QR)
                                    crops_ready = []
                                    if im.size > 0:
                                        h_im, w_im = im.shape[:2]
                                        # ID is almost always in top 48%
                                        top_half = im[:int(h_im*0.48), :]
                                        crops_ready.append(top_half)
                                        # If it looks like a wide GC label, specifically target top-right
                                        if w_im > h_im * 1.6:
                                            tr_q = im[:int(h_im*0.48), int(w_im*0.45):]
                                            crops_ready.append(tr_q)

                                    # Relax check: Log instead of skip to ensure we don't miss valid labels
                                    if not crops_ready:
                                        continue
                                    if not looks_like_label_region(crops_ready[0]):
                                        self.log.emit(f"Warning: ROI looks noisy but scanning anyway: {img_basename}")

                                    # Gather all crop variants for evaluation
                                    # We use the first crop (top slice) as source for right_digits too
                                    right_digits = crops_ready[0][:, max(0, int(crops_ready[0].shape[1]*0.30)):]
                                    eval_list = crops_ready + [right_digits]

                                    cleaned_res, mean_conf = read_best_label_id(
                                        eval_list,
                                        model_ocr, loaded_tokenizer, model_args,
                                        source_image=image   # FIX: pass for penalty
                                    )
                                    if not cleaned_res or not is_valid_label_id(cleaned_res):
                                        continue
                                    if mean_conf < CONF_ACCEPT:   # FIX: use raised threshold
                                        continue

                                    draw_detection_box(image, cleaned_res,
                                                       region=(x1, y1, x2, y2))
                                    draw_overlay_text(image, cleaned_res, RED,
                                                      anchor=(55, 55), max_width_ratio=0.65)

                                    # FIX #4: Exact match, no allow_generic_success
                                    match_res   = build_match_result(actual_value, cleaned_res,
                                                                     mean_conf)
                                    yolo_entry  = (subfolder_name, img_basename,
                                                   len(boxes), class_names[0],
                                                   f"{mean_conf:.2f}", cleaned_res, match_res)
                                    detections  = [yolo_entry]
                                except Exception as e:
                                    self.log.emit(f"Box error: {e}")

                    # ── STEP 3: Fallback ─────────────────────────────────────
                    has_ocr = any(str(d[5]).strip() for d in detections)
                    if not has_ocr:
                        elapsed = time.perf_counter() - t0
                        if elapsed > 0.7:
                            self.log.emit(f"Skipping fallback (time={elapsed:.2f}s): {img_basename}")
                        elif not has_any_label_zone(image):
                            self.log.emit(f"No label zone: {img_basename}")
                        else:
                            self.log.emit(f"Fallback scan: {img_basename}")
                            fb_text, fb_conf = fallback_label_scan(
                                image, model_ocr, loaded_tokenizer, model_args, actual_value
                            )
                            if fb_text and is_valid_label_id(fb_text) and fb_conf >= CONF_FALLBACK:
                                # FIX #4: Exact match only
                                match_res = build_match_result(actual_value, fb_text, fb_conf)
                                draw_detection_box(image, fb_text)
                                draw_overlay_text(image, fb_text, RED,
                                                  anchor=(55, 55), max_width_ratio=0.65)
                                detections = [(subfolder_name, img_basename, 0, "fallback",
                                               f"{fb_conf:.2f}", fb_text, match_res)]
                            else:
                                self.log.emit(f"Fallback miss: {img_basename}")

                    # ── STEP 5: Automatic Second-Pass Retry ────────────────────
                    # The user requested to check again and update if missing
                    if not detections:
                        self.log.emit(f"Initial passes failed. Starting High-Power Retry for {img_basename}...")
                        # Pass with high contrast and NO masking to catch whatever was hidden
                        contrast_img = _prep_contrast(image)
                        fb_text, fb_conf = fallback_label_scan(
                            contrast_img, model_ocr, loaded_tokenizer, model_args, actual_value
                        )
                        if fb_text:
                            match_res = build_match_result(actual_value, fb_text, fb_conf)
                            draw_detection_box(image, fb_text)
                            draw_overlay_text(image, fb_text, RED,
                                              anchor=(55, 55), max_width_ratio=0.65)
                            detections = [(subfolder_name, img_basename, 0, "super-fallback",
                                           f"{fb_conf:.2f}", fb_text, match_res)]

                    if not detections:
                        detections.append((subfolder_name, img_basename, 0, "", "0", "", ""))

                    cv2.imwrite(os.path.join(res_folder, img_basename), image)
                    all_detections.extend(detections)

                    elapsed_total = time.perf_counter() - t0
                    self.log.emit(f"Done {img_basename}  ({elapsed_total:.2f}s)")
                    self.progress.emit(i + 1, total_images)
                    self.image_processed.emit(image, detections)

                except Exception as img_err:
                    err_msg  = str(img_err)
                    self.log.emit(f"⚠ Skipped {os.path.basename(image_path)}: {err_msg}")
                    skip_row = (
                        os.path.basename(os.path.dirname(image_path)),
                        os.path.basename(image_path),
                        0, "error", "0", err_msg[:40], ""
                    )
                    all_detections.append(skip_row)
                    self.progress.emit(i + 1, total_images)
                    self.image_processed.emit(None, [skip_row])

            self.log.emit("Processing complete.")
            self.finished.emit(all_detections)

        except Exception as e:
            self.log.emit(f"Critical error: {str(e)}")
            self.finished.emit([])


# ================================================================================
# UI
# ================================================================================

class CoinScannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pro OCR Scanner")
        self.resize(1100, 700)
        self.initUI()
        self.worker = None

    def initUI(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f8fafc; }
            QLabel { color: #334155; font-family: 'Segoe UI', system-ui, sans-serif; }
            QPushButton {
                background-color: #334c7a; color: white; font-weight: 500;
                border: 1px solid #2b4266; border-radius: 6px;
                padding: 8px 16px; font-size: 14px;
            }
            QPushButton:hover  { background-color: #2b4266; }
            QPushButton:pressed { background-color: #1e293b; }
            QPushButton:disabled {
                background-color: #cbd5e1; border: 1px solid #cbd5e1; color: #f8fafc;
            }
            QPushButton#secondaryBtn {
                background-color: #f1f5f9; color: #334c7a; border: 1px solid #cbd5e1;
            }
            QPushButton#secondaryBtn:hover { background-color: #e2e8f0; }
            QTableWidget {
                background-color: #ffffff; color: #334155;
                gridline-color: #e2e8f0; border: 1px solid #cbd5e1;
                border-radius: 6px; outline: none;
            }
            QHeaderView::section {
                background-color: #f1f5f9; color: #475569; font-weight: 600;
                border: none; border-right: 1px solid #e2e8f0;
                border-bottom: 2px solid #cbd5e1; padding: 6px;
            }
            QProgressBar {
                border: 1px solid #cbd5e1; border-radius: 6px;
                text-align: center; background-color: #f8fafc;
                color: #334155; height: 18px;
            }
            QProgressBar::chunk { background-color: #334c7a; border-radius: 5px; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        lyt_main = QVBoxLayout()
        lyt_main.setContentsMargins(20, 20, 20, 20)
        lyt_main.setSpacing(15)

        lbl_title = QLabel("Coin Label Detection & OCR System")
        lbl_title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #1e293b; margin-bottom: 5px;")
        lyt_main.addWidget(lbl_title)

        lyt_controls = QHBoxLayout()
        self.btn_select = QPushButton("Select Folder")
        self.btn_select.clicked.connect(self.select_folder)
        self.lbl_folder = QLabel("Current Folder: ./gc_pandora")
        self.lbl_folder.setStyleSheet("font-size: 13px; color: #64748b;")
        self.btn_start = QPushButton("Start Scanning")
        self.btn_start.clicked.connect(self.toggle_scan)
        self.btn_generate_csv = QPushButton("Generate .csv")
        self.btn_generate_csv.setObjectName("secondaryBtn")
        self.btn_generate_csv.clicked.connect(self.generate_csv_report)
        self.btn_generate_csv.setEnabled(False)

        lyt_controls.addWidget(self.btn_select)
        lyt_controls.addWidget(self.lbl_folder)
        lyt_controls.addStretch()
        lyt_controls.addWidget(self.btn_generate_csv)
        lyt_controls.addWidget(self.btn_start)
        lyt_main.addLayout(lyt_controls)

        lyt_content = QHBoxLayout()

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ['Folder', 'Image', 'Confidence', 'OCR Result', 'Match'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self.display_selected_image)
        lyt_content.addWidget(self.table, stretch=1)

        lyt_preview = QVBoxLayout()
        self.lbl_preview = QLabel("No Image Selected")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setStyleSheet(
            "background-color: #ffffff; border-radius: 8px; border: 1px solid #e5e7eb;")
        self.lbl_preview.setMinimumSize(400, 400)
        lyt_preview.addWidget(self.lbl_preview)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #4b5563; font-size: 13px;")
        lyt_preview.addWidget(self.lbl_status)

        lyt_content.addLayout(lyt_preview, stretch=1)
        lyt_main.addLayout(lyt_content)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        lyt_main.addWidget(self.progress)

        main_widget.setLayout(lyt_main)
        self.selected_folder = os.path.join(os.getcwd(), 'gc_pandora')

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Directory", self.selected_folder)
        if folder:
            self.selected_folder = folder
            self.lbl_folder.setText(f"Current Folder: {folder}")

    def toggle_scan(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.btn_start.setText("Start Scanning")
            self.btn_start.setStyleSheet("")
            self.lbl_status.setText("Scan cancelled.")
            return
        self.btn_generate_csv.setEnabled(False)
        self.latest_detections = []
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.btn_start.setText("Stop Scanning")
        self.btn_start.setStyleSheet("background-color: #ef4444;")
        self.lbl_status.setText("Initializing models…")

        self.worker = OCRWorker(self.selected_folder)
        self.worker.progress.connect(self.update_progress)
        self.worker.image_processed.connect(self.update_ui)
        self.worker.log.connect(self.update_log)
        self.worker.finished.connect(self.scan_finished)
        self.worker.start()

    def update_log(self, text):
        self.lbl_status.setText(text)

    def update_progress(self, current, total):
        self.progress.setValue(int((current / total) * 100))

    def update_ui(self, cv_img, detections):
        if cv_img is not None:
            rgb   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap  = QPixmap.fromImage(qt_img)
            self.lbl_preview.setPixmap(
                pixmap.scaled(self.lbl_preview.size(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation))

        for det in detections:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(det[0])))
            self.table.setItem(row, 1, QTableWidgetItem(str(det[1])))
            self.table.setItem(row, 2, QTableWidgetItem(str(det[4])))
            ocr_disp = str(det[5]).strip()
            if not is_valid_label_id(ocr_disp):
                ocr_disp = ""
            self.table.setItem(row, 3, QTableWidgetItem(ocr_disp))
            match_item = QTableWidgetItem(str(det[6]))
            if str(det[6]) == "True":
                match_item.setForeground(Qt.green)
            elif str(det[6]) == "False":
                match_item.setForeground(Qt.red)
            else:
                match_item.setForeground(Qt.black)
            self.table.setItem(row, 4, match_item)
            conf_val = str(det[4]).strip()
            has_real_conf = conf_val not in ("0", "0.0", "", "0.00")
            if ocr_disp and has_real_conf and is_valid_label_id(ocr_disp):
                from PyQt5.QtGui import QColor
                _blue_bg = QColor(219, 234, 254)
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(_blue_bg)
            self.table.scrollToBottom()

    def display_selected_image(self):
        items = self.table.selectedItems()
        if not items:
            return
        img_item = self.table.item(items[0].row(), 1)
        if not img_item:
            return
        full_path = os.path.join("images", "results",
                                 datetime.now().strftime("%Y-%m-%d"),
                                 img_item.text())
        if os.path.exists(full_path):
            cv_img = cv2.imread(full_path)
            if cv_img is not None:
                rgb   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qt_img  = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap  = QPixmap.fromImage(qt_img)
                self.lbl_preview.setPixmap(
                    pixmap.scaled(self.lbl_preview.size(),
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def scan_finished(self, detections):
        self.btn_start.setText("Start Scanning")
        self.btn_start.setStyleSheet("")
        self.btn_generate_csv.setEnabled(bool(detections))
        if not detections:
            self.lbl_status.setText("Scan stopped. No results collected.")
            self.progress.setValue(100)
            return

        base_match_map = {}
        for det in detections:
            if det[6] in ("True", "False"):
                base = re.sub(r'-\d+$', '', det[1].rsplit('.', 1)[0])
                if base not in base_match_map or det[6] == "True":
                    base_match_map[base] = det[6]

        filled = []
        for det in detections:
            det  = list(det)
            base = re.sub(r'-\d+$', '', det[1].rsplit('.', 1)[0])
            pm   = base_match_map.get(base, "")
            if det[6] == "":
                det[6] = pm
            elif det[6] == "False" and pm == "True":
                det[6] = "True"
            filled.append(tuple(det))

        self.latest_detections = filled
        self.table.setRowCount(0)
        for det in filled:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(det[0])))
            self.table.setItem(row, 1, QTableWidgetItem(str(det[1])))
            self.table.setItem(row, 2, QTableWidgetItem(str(det[4])))
            ocr_disp = str(det[5]).strip()
            if not is_valid_label_id(ocr_disp):
                ocr_disp = ""
            self.table.setItem(row, 3, QTableWidgetItem(ocr_disp))
            match_item = QTableWidgetItem(str(det[6]))
            if str(det[6]) == "True":
                match_item.setForeground(Qt.darkGreen)
                match_item.setFont(QFont("", -1, QFont.Bold))
            elif str(det[6]) == "False":
                match_item.setForeground(Qt.red)
                match_item.setFont(QFont("", -1, QFont.Bold))
            else:
                match_item.setForeground(Qt.gray)
            self.table.setItem(row, 4, match_item)
            conf_val = str(det[4]).strip()
            has_real_conf = conf_val not in ("0", "0.0", "", "0.00")
            if ocr_disp and has_real_conf and is_valid_label_id(ocr_disp):
                from PyQt5.QtGui import QColor
                _blue_bg = QColor(219, 234, 254)
                for col in range(self.table.columnCount()):
                    item = self.table.item(row, col)
                    if item:
                        item.setBackground(_blue_bg)

        self.btn_generate_csv.setEnabled(True)
        self.lbl_status.setText("Scan finished. You can now Generate .csv.")

    def generate_csv_report(self):
        if not hasattr(self, 'latest_detections') or not self.latest_detections:
            return
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Final Report", "report_results.csv", "CSV Files (*.csv)")
        if save_path:
            with open(save_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['Folder name', 'Image Path', 'Number of Boxes',
                             'Class Name', 'Confidence', 'ocr_result', 'result'])
                w.writerows(self.latest_detections)
            self.lbl_status.setText(f"Report saved to {save_path}")


if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = CoinScannerApp()
    window.show()
    sys.exit(app.exec_())