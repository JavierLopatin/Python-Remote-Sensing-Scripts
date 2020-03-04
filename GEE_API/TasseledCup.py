### simple example to get Tasseled cup values and visualization

from IPython.display import Image
import ee
import VI_Landsat as vi

ee.Initialize()

# Input Landsat 8 TOA image in Oaxaca, Veracruz, Tabasco and Chiapas México
image = ee.Image('LANDSAT/LC8_L1T_TOA/LC80230482016032LGN00')
region = image.geometry().getInfo()['coordinates']

# Execute 'Tasseled cap Transformation'
tasseled_cap = vi.tasseled_cap_transformation(image)
tasseled_cap.bandNames().getInfo()

# Display result
thumbnail = tasseled_cap.getThumbUrl({'min': -1, 'max': 1, 'region': region,
    'bands': 'brightness,fourth,sixth'}) # default RGB always b1,b2,b3
Image(url=thumbnail)
