from PIL import Image, ImageDraw, ImageFont

img = Image.open('/Users/hangquanqian/desktop/tt.jpg')
draw = ImageDraw.Draw(img)
myfont = ImageFont.truetype('Arial.ttf',size=30)
fillcolor = "#ff0000"
width,height=img.size
draw.text((40,40),'Tao Mamba', font=myfont,fill=fillcolor)
img.save('tt1.png')


