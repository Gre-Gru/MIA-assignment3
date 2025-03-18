import os

import SimpleITK as sitk
import numpy as np


if __name__ == '__main__':

    # Download the file "assignment3-data.zip" from the Teach Center and unpack it in the main folder of your repo
    # Do not commit any data files for this assignment, since these are too large in size for checking into git repos
    data_base_path = "./data"

    modality = 'ct'
    # original data - CT or MR
    images_dir = os.path.join(data_base_path, "img")
    labels_dir = os.path.join(data_base_path, "seg")

    # this looks for commonly used file endings for volumetric files (MetaImage, Nifti or Nrrd)
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.mha') or f.endswith('.nii') or f.endswith('.nrrd')]
    image_files.sort(reverse=False)

    print(image_files)

    # This example only considers the first of the 20 available input images
    image_filename = image_files[0]
    identifier = image_filename.split('_image_isotropic')[0]
    label_filename = f"{identifier}_label_isotropic.mha"
    image_path = os.path.join(images_dir, image_filename)
    label_path = os.path.join(labels_dir, label_filename)

    label_image = sitk.ReadImage(label_path)
    input_image = sitk.ReadImage(image_path)

    print("data type of input image: ", input_image.GetPixelIDTypeAsString())
    print("data type of label image: ", label_image.GetPixelIDTypeAsString())

    # TAKE CARE: Simple ITK and numpy arrays interpret x-y-z dimensions differently
    # numpy swaps x and z dimension, this has to be taken into account when switching between Simple ITK and numpy!
    # see also: https://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html
    input_image_as_np_array = sitk.GetArrayFromImage(input_image)

    min_greyval = np.min(input_image_as_np_array)
    max_greyval = np.max(input_image_as_np_array)

    # print some important attributes of a 3D volumetric image
    print("img filename, image size, image spacing, image origin, image orientation, image min greyvalue, image max greyvalue:")
    print(image_filename, input_image.GetSize(), input_image.GetSpacing(), input_image.GetOrigin(),
          input_image.GetDirection(), min_greyval, max_greyval)

    # manipulate image intensities via numpy (our example only manipulates intensities, so we don't have to
    # consider different x-y-z indexing, but as soon as you compute (average) locations like center of gravity,
    # getting x-y-z indexing right is very important!
    input_image_as_np_array = (input_image_as_np_array - min_greyval)
    # this inverts the intensities in the signed integer range between original min and max greyvalues
    input_image_as_np_array = max_greyval - input_image_as_np_array

    # now we convert back from numpy array to sitk image, so we can use the sitk write functionality
    modified_image = sitk.GetImageFromArray(input_image_as_np_array)

    output_filename = image_filename.replace(".mha", "_modified.mha")
    sitk.WriteImage(modified_image, output_filename, useCompression=True)
