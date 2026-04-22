import sys

with open('/opt/lampp/htdocs/ocr/app.py', 'r') as f:
    content = f.read()

# Replace YOLO imwrite
content = content.replace("cv2.putText(image, str(cleaned_res), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)\n                            cv2.imwrite(res_folder_name + \"/\" + img_basename, image)", "cv2.putText(image, str(cleaned_res), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)")

# Replace fallback imwrite
content = content.replace("cv2.putText(image, str(ocr_res), (60, 80),\n                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 0), 2)\n                            cv2.imwrite(res_folder_name + \"/\" + img_basename, image)", "cv2.putText(image, str(ocr_res), (60, 80),\n                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 0), 2)")

# Add unconditionally saving image at the end of the loop
replacement_end = """                if not detections:
                    detections.append((subfolder_name, os.path.basename(image_path), 0, "", "0", "", ""))

                # Always save the image at the end so stale images from previous runs are overwritten
                cv2.imwrite(res_folder_name + "/" + img_basename, image)

                all_detections.extend(detections)"""

content = content.replace("                if not detections:\n                    detections.append((subfolder_name, os.path.basename(image_path), 0, \"\", \"0\", \"\", \"\"))\n\n                all_detections.extend(detections)", replacement_end)

with open('/opt/lampp/htdocs/ocr/app.py', 'w') as f:
    f.write(content)
