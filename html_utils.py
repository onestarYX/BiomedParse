def write_start(file):
    file.write("<!DOCTYPE html>\n<html><body>\n")

def write_end(file):
    file.write("</body></html>")

def write_head(file, level, msg):
    file.write(f"<h{level}>{msg}</h{level}>\n")

def write_text(file, msg):
    file.write(f"<p>{msg}")
    # file.write(msg)

def write_img(file, img_path, caption):
    file.write(f'<img src="{img_path}" alt="{caption}">\n')