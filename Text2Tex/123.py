from PIL import Image

import imageio
import numpy as np
import pdb

a = Image.open("./samples/textures/dummy.png").convert("RGB").resize((3000, 3000))

pdb.set_trace()