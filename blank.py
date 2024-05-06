import os
from PIL import Image

num_images = 100

directory = 'English/Fnt/Sample063'

os.makedirs(directory, exist_ok=True)

image_size = (128, 128)
background_color = 'white'  

for i in range(num_images):
    img = Image.new('RGB', image_size, color=background_color)
    filename = f'blank_{i+1}.png'
    img.save(os.path.join(directory, filename))

print(f'Successfully created and saved {num_images} blank images in {directory}.')
