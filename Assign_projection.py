#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add missing projection information to GDAL dataset from:

* Existing image
* WKT file
* Proj4 string
* EPSG code

For where GDAL dataset has correct geoinformation but projection
has not been stored correctly.

For example when exporting files from APL using projections other than
Lat/Long WGS84, UTM or OSGB36 NG this script can be used to fix the
following error:

"Unknown map projection: You will have to fill in the projection name 
                and datum in the map info in the .hdr file yourself."

Note this script will not reproject data. For this use gdalwarp

Requires GDAL Python bindings to be installed and available

"""
###########################################################
# This file has been created by ARSF Data Analysis Node and
# is licensed under the GPL v3 Licence. A copy of this
# licence is available to download with this file.
###########################################################

###########################################################
# Use like, e.g.:
# to assign a projection from a WKT file:
# python assign_projection.py --wkt 'UTM50S.wkt' image_without_projection.kea
# If you have a lot of files which need the projection assigning you can use:
# python assign_projection.py --wkt 'UTM50S' *.kea
# As well as a WKT file you can also use Proj4 strings, for example:
# python assign_projection.py \
# --proj4 '+proj=utm +zone=55 +south +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs' \
#  image_without_projection.kea
###########################################################

import argparse
import sys
import glob
try:
   from osgeo import gdal, osr
except ImportError:
   print('ERROR: Could not import GDAL Python library.')
   print('Check GDAL is installed and if necessary added to "PYTHONPATH" environmental variable')
   sys.exit(2)

def assign_from_image(in_image, in_basefile):
   """
   Assign projection to one GDAL dataset
   from another.

   Arguments:

   * in_image - Input image to assign projection to
   * in_basefile - Input image to copy projection information from

   """
   print('Copying projection from {} to {}'.format(in_basefile, in_image))
   # Open the image file, in update mode
   # so that the image can be edited. 
   baseDS = gdal.Open(in_basefile, gdal.GA_ReadOnly)
   outputDS = gdal.Open(in_image, gdal.GA_Update)
   
   if (not baseDS is None) and (not outputDS is None):
       outputDS.SetProjection(baseDS.GetProjection())
   else:
       raise Exception("Could not open input / output file")
   
   baseDS = None
   outputDS = None

def assign_from_wkt(in_image, in_wkt_file):
   """
   Assign projection to a GDAL dataset
   from WKT file

   Arguments:

   * in_image - Input image to assign projection to
   * in_wkt_file - Input WKT file

   """
   print('Adding projection from {} to {}'.format(in_wkt_file, in_image))

   # Read WKT file to a string
   in_wkt_handler = open(in_wkt_file,'r')
   in_wkt_str = in_wkt_handler.read()
   in_wkt_handler.close()

   # Open the image file, in update mode
   # so that the image can be edited. 
   outputDS = gdal.Open(in_image, gdal.GA_Update)
   
   if not outputDS is None:
       outputDS.SetProjection(in_wkt_str)
   else:
       raise Exception("Could not open input / output file")
   
   outputDS = None

def assign_from_proj4(in_image, in_proj4):
   """
   Assign projection to a GDAL dataset
   from Proj4 string

   Arguments:

   * in_image - Input image to assign projection to
   * in_proj4 - Proj4 string

   """
   print('Adding projection {} to {}'.format(in_proj4, in_image))

   # Check if Proj4 string is within quotes
   # remove if so.
   if in_proj4[0] == '"' or in_proj4[0] == '\'':
      in_proj4 = in_proj4[1:-1]

   spatial_ref = osr.SpatialReference()
   osr_out = spatial_ref.ImportFromProj4(in_proj4)

   if osr_out != 0:
      raise Exception('Could not create projection. '\
                   'Is "{}" a valid proj4 string'.format(in_proj4))

   wkt_str = spatial_ref.ExportToWkt()

   # Open the image file, in update mode
   # so that the image can be edited. 
   outputDS = gdal.Open(in_image, gdal.GA_Update)
   
   if not outputDS is None:
       outputDS.SetProjection(wkt_str)
   else:
       raise Exception("Could not open input / output file")
   
   outputDS = None

def assign_from_epsg_code(in_image, epsg_code):
   """
   Assign projection to a GDAL dataset
   from EPSG code

   Arguments:

   * in_image - Input image to assign projection to
   * epsg_code - EPSG code 

   """
   # Tidy up EPSG code format
   if isinstance(epsg_code,int):
      epsg_code_int = epsg_code
   elif isinstance(epsg_code,str):
      epsg_code_int = int(epsg_code.split(':')[-1])
   else:
      raise Exception('Did not understand input format, expected string or integer')

   print('Adding projection from EPSG code {} to {}'.format(epsg_code, in_image))

   spatial_ref = osr.SpatialReference()
   osr_out = spatial_ref.ImportFromEPSG(epsg_code_int)

   if osr_out != 0:
      raise Exception('Could not create projection. '\
                   'Is "{}" a valid EPGS code'.format(epsg_code_int))

   wkt_str = spatial_ref.ExportToWkt()

   # Open the image file, in update mode
   # so that the image can be edited. 
   outputDS = gdal.Open(in_image, gdal.GA_Update)
   
   if not outputDS is None:
       outputDS.SetProjection(wkt_str)
   else:
       raise Exception("Could not open input / output file")
   
   outputDS = None

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='''Assign missing projection information to a GDAL dataset. 
                                             Note this script does not reproject data it 
                                             only adds an existing projection in a format GDAL can recognise. 
                                             Created by ARSF-DAN at Plymouth Marine Laboratory. 
                                             Latest version available from https://github.com/pmlrsg/arsf_tools/.''')
   parser.add_argument("inputfile", nargs='+',type=str, help="Input Files(s)")
   parser.add_argument("-b", "--baseimage", type=str, 
                       help="Copy projection information from specified image", 
                       required=False,
                       default=None)
   parser.add_argument("-w", "--wkt", type=str, 
                       help="Assign projection information from specified WKT file", 
                       required=False,
                       default=None)
   parser.add_argument("-p", "--proj4", type=str, 
                       help="Assign projection information from specified Proj4 string", 
                       required=False,
                       default=None)
   parser.add_argument("-e", "--epsg", type=str, 
                       help="Assign projection information from EPSG code", 
                       required=False,
                       default=None)
   args = parser.parse_args()

   # On Windows don't have shell expansion so fake it using glob
   if args.inputfile[0].find('*') > -1:
      input_file_list = glob.glob(args.inputfile[0])   
   else:
      input_file_list = args.inputfile

   try:
      for image in input_file_list:
         if args.baseimage is not None:
            assign_from_image(image, args.baseimage)
         elif args.wkt is not None:
            assign_from_wkt(image, args.wkt)
         elif args.proj4 is not None:
            assign_from_proj4(image, args.proj4)
         elif args.epsg is not None:
            assign_from_epsg_code(image, args.epsg)
         else:
            print("ERROR: must provide one of '--baseimage','--wkt','--proj4' or '--epsg'")
            sys.exit(1)
   except Exception as err:
      print(err)
      sys.exit(1)

