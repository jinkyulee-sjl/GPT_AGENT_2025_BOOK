import pymupdf
import os

base_name_1 = "Hands-On Machine Learning Print.pdf"
base_name_2 = "Calculus Scanned Print.pdf"
base_name_3 = "견적서 1.pdf"
pdf_file_path = f"chap04/data/{base_name_3}"
doc = pymupdf.open(pdf_file_path)

header_height = 80  # 헤더 높이 설정
footer_height = 80  # 푸터 높이 설정

full_text = ""

for page in doc:
    # page_height = page.rect.height
    # text = page.get_textbox(pymupdf.Rect(0, header_heiht, page.rect.width, page_height - footer_height))
    # full_text += text 
    rect = page.rect
    header = page.get_text(clip=(0, 0, rect.width, header_height))
    footer = page.get_text(clip=(0, rect.height - footer_height, rect.width, rect.height))
    text = page.get_text(clip=(0, header_height, rect.width, rect.height - footer_height))

    full_text += text + "\n---------------------------\n"

# 파일명만 추출
pdf_file_name = os.path.basename(pdf_file_path)
pdf_file_path = os.path.splitext(pdf_file_name)[0]  # 확장자 제거

txt_file_path = f"chap04/output/{pdf_file_path}_with_preprocessing.txt"

with open(txt_file_path, "w", encoding="utf-8") as f:
    f.write(full_text)