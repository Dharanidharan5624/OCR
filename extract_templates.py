import cv2
import numpy as np
import os

def extract_templates(image_path, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not read image.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of digits
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and sort contours from left to right
    digit_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 20 < h < 100 and 5 < w < 80:
            digit_contours.append((x, y, w, h))
    
    digit_contours.sort(key=lambda c: c[0])
    
    if len(digit_contours) != 10:
        print(f"Warning: Found {len(digit_contours)} potential digits instead of 10.")
        # We'll take the first 10 for now
        digit_contours = digit_contours[:10]
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, (x, y, w, h) in enumerate(digit_contours):
        digit = img[y:y+h, x:x+w]
        template_path = os.path.join(output_dir, f"{i}.png")
        cv2.imwrite(template_path, digit)
        print(f"Saved template {i} to {template_path}")

if __name__ == "__main__":
    extract_templates("Lable.png", "font_templates")
