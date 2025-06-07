import easyocr

def recognize_text(image):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image, detail=0, paragraph=True)
    return result