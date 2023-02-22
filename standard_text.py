import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Std_Text(object):
    def __init__(self, font_path):
        self.height = 64
        self.max_width = 960
        self.border_width = 5
        self.char_list = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        font_height = self.get_valid_height(font_path)
        self.font = ImageFont.truetype(font_path, font_height)

    def get_valid_height(self, font_path):
        font = ImageFont.truetype(font_path, self.height - 4)
        _, font_height = font.getsize(self.char_list)
        if font_height <= self.height - 4:
            return self.height - 4
        else:
            return int((self.height - 4)**2 / font_height)

    def draw_text(self, text):
        assert len(text) != 0

        char_x = self.border_width
        bg = Image.new("RGB", (self.max_width, self.height), color=(127, 127, 127))
        draw = ImageDraw.Draw(bg)
        for char in text:
            draw.text((char_x, 2), char, fill=(0, 0, 0), font=self.font)
            char_size = self.font.getsize(char)[0]
            char_x += char_size

        canvas = np.array(bg).astype(np.uint8)
        char_x += self.border_width
        canvas = canvas[:, :char_x, :]

        return canvas


def main():
    font_path = 'arial.ttf'
    std_text = Std_Text(font_path)

    tmp = std_text.draw_text('qwertyuiopasdfghjklzxcvbnm')
    print(tmp.shape)
    cv2.imwrite('tmp.jpg', tmp)


if __name__ == '__main__':
    main()
