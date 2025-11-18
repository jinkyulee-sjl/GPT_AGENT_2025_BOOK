# PDF 파일을 텍스트 파일로 변환하는 스크립트

import pymupdf
import os

base_name_1 = "Hands-On Machine Learning Print.pdf"
base_name_2 = "Calculus Scanned Print.pdf"
base_name_3 = "견적서 1.pdf"

pdf_file_path = f"chap04/data/{base_name_1}"
doc = pymupdf.open(pdf_file_path)

full_text = ""

for page in doc:
    text = page.get_text()
    full_text += text

pdf_file_name = os.path.basename(pdf_file_path)
pdf_file_path = os.path.splitext(pdf_file_name)[0]  # 확장자 제거

txt_file_path = f"chap04/output/{pdf_file_path}.txt"

with open(txt_file_path, "w", encoding="utf-8") as f:
    f.write(full_text)

