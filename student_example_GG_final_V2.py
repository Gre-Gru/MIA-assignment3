import os

import SimpleITK as sitk
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def label_cog(label_image):
    label_array = sitk.GetArrayFromImage(label_image) # in Numpy
    labels = np.unique(label_array)

    #print(labels) # 0 = background

    centroids = {}
    for label in labels:
        if label == 0:  
            continue

        mask = label_array == label
        indices = np.argwhere(mask)
        centroid = np.mean(indices, axis=0)
        centroid_world = label_image.TransformContinuousIndexToPhysicalPoint(centroid[::-1]) # back to SiTK
        centroids[label] = centroid_world 

    return centroids

def visualize_3d_points(source_points, target_points, source_mean, target_mean):

    source_np = np.array(source_points) 
    target_np = np.array(target_points)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(source_np[:, 0], source_np[:, 1], source_np[:, 2], color='blue', marker='o', label='Source COGs')
    ax.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], color='red', marker='^', label='Target COGs')
    ax.scatter(source_mean[0], source_mean[1], source_mean[2], color='blue', marker='x', s=200, label='Source rotation center')
    ax.scatter(target_mean[0], target_mean[1], target_mean[2], color='red', marker='x', s=200, label='Target rotation center')

    for i in range(len(source_np)):
        ax.plot([source_np[i, 0], target_np[i, 0]], 
                [source_np[i, 1], target_np[i, 1]], 
                [source_np[i, 2], target_np[i, 2]], 'gray', linestyle='dotted')

    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Visualization of Source and Target COGs")
    ax.legend()

    label_text = "Center of rotation:\n"
    label_text += f"Source: ({source_mean[0]:.1f}, {source_mean[1]:.1f}, {source_mean[2]:.1f})\n"
    label_text += f"Target: ({target_mean[0]:.1f}, {target_mean[1]:.1f}, {target_mean[2]:.1f})\n"
 
    plt.figtext(0.15, 0.02, label_text, wrap=True, horizontalalignment='left', fontsize=10)
    plt.savefig(f'3D_plot_{source_mean[0]:.1f}.png')
    plt.show()

def procrustes(source_points, target_points):
    source = np.array(source_points)
    target = np.array(target_points)

    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    #visualize_3d_points(source_points, target_points, source_mean, target_mean)

    P = source - source_mean 
    Q = target - target_mean 

    H = np.dot(Q.T, P)

    U, S, Vt = svd(H)
    R = np.dot(U, Vt)  # Rotation matrix

    #print(f'Center source: {P}')
    #print(f'Center target: {Q}')

    scale_factor = np.linalg.norm(Q) / np.linalg.norm(P)

    #T = target_mean - scale_factor * np.dot(R, source_mean)
    T = target_mean - source_mean

    return R, T, scale_factor

def apply_transformation(input_image, R, T, scale_factor, reference_image, center):
        
    #Similarity Transform
    scaled_R = scale_factor * R.T
    transform = sitk.Similarity3DTransform()
    transform.SetCenter(center)
    transform.SetMatrix(scaled_R.flatten())
    transform.SetTranslation(-T)

    #Affine Transform 
    #R = R.T
    # transform = sitk.AffineTransform(3)
    #transform.SetCenter(center)
    # transform.SetMatrix(R.flatten())
    # transform.Scale(scale_factor)
    # transform.SetTranslation(-T)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)

    resampled_image = resampler.Execute(input_image)
    return resampled_image

def transform_label(label, R, T, scale_factor, reference_label, center):
    scaled_R = scale_factor * R.T
    transform = sitk.Similarity3DTransform()
    transform.SetCenter(center)
    transform.SetMatrix(scaled_R.flatten())
    transform.SetTranslation(-T)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_label)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(transform)

    resampled_image = resampler.Execute(label)
    return resampled_image


def compute_average_volume(images):
    sum_array = None
    for image in images:
        image_array = sitk.GetArrayFromImage(image)
        if sum_array is None:
            sum_array = np.zeros_like(image_array, dtype=np.float64)
        sum_array += image_array

    avg_array = sum_array / len(images)
    average_image = sitk.GetImageFromArray(avg_array)
    return average_image

def compute_centroids(label_file):
    
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(label_file)
    
    centroids = {}
    
    for label in range(1, 9):
        if label_stats.HasLabel(label):
            centroids[label] = label_stats.GetCentroid(label)
    
    return centroids

def crop_volume(image, label, crop_size=(128, 128, 128)):

    label_array = sitk.GetArrayFromImage(label)
    z, y, x = np.where(label_array > 0)

    center_x = (x.min() + x.max()) // 2
    center_y = (y.min() + y.max()) // 2
    center_z = (z.min() + z.max()) // 2

    #print(x.min(), x.max(), y.min(), y.max(), z.min(), z.max())

    start_x = max(center_x - crop_size[0] // 2, 0)
    start_y = max(center_y - crop_size[1] // 2, 0)
    start_z = max(center_z - crop_size[2] // 2, 0)

    end_x = min(start_x + crop_size[0], label.GetWidth())
    end_y = min(start_y + crop_size[1], label.GetHeight())
    end_z = min(start_z + crop_size[2], label.GetDepth())

    start_x = max(end_x - crop_size[0], 0)
    start_y = max(end_y - crop_size[1], 0)
    start_z = max(end_z - crop_size[2], 0)

    roi_start = [int(start_x), int(start_y), int(start_z)]
    roi_size = list(crop_size)
    
    cropped_image = sitk.RegionOfInterest(image, roi_size, roi_start)
    cropped_label = sitk.RegionOfInterest(label, roi_size, roi_start)
    
    return cropped_image, cropped_label


if __name__ == '__main__':

    # Download the file "assignment3-data.zip" from the Teach Center and unpack it in the main folder of your repo
    # Do not commit any data files for this assignment, since these are too large in size for checking into git repos
    data_base_path = "./data"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    modality = 'ct'
    # original data - CT or MR
    images_dir = os.path.join(data_base_path, "img")
    labels_dir = os.path.join(data_base_path, "seg")

    # this looks for commonly used file endings for volumetric files (MetaImage, Nifti or Nrrd)
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.mha') or f.endswith('.nii') or f.endswith('.nrrd')]
    image_files.sort(reverse=False)

    #print(image_files)
    resampled_images, resampled_labels, images, labels = [], [], [], []

    for image_filename in image_files:
        identifier = image_filename.split('_image_isotropic')[0]
        label_filename = f"{identifier}_label_isotropic.mha"
        images.append(sitk.ReadImage(os.path.join(images_dir, image_filename)))
        labels.append(sitk.ReadImage(os.path.join(labels_dir, label_filename)))

    average_before_reg = compute_average_volume(images)
    reference_index = 0
    reference_image_filename = image_files[reference_index]
    reference_label = labels[reference_index]
    ref_centroids = label_cog(reference_label)

    centroids_test = compute_centroids(labels[reference_index])
    # print(ref_centroids)
    # print(centroids_test)

    for i, (image, label) in enumerate(zip(images, labels)):
        if i == reference_index:
            resampled_images.append(image)
            continue

        centroids = label_cog(label)
      
        R, T, scale_factor = procrustes(list(centroids.values()), list(ref_centroids.values()))

        cent = np.mean(list(centroids.values()), axis=0)

        resampled_image = apply_transformation(image, R, T, scale_factor, reference_label, cent)
        resampled_images.append(resampled_image)

        resampled_label = transform_label(label, R, T, scale_factor, reference_label, cent)
        resampled_labels.append(resampled_label)

        if i == 1:
            output_filename = os.path.join(output_dir, f"registered_volume_{i+1}.mha")
            sitk.WriteImage(resampled_image, output_filename)


    cropped_images, cropped_labels = [], []
    for image, label in zip(resampled_images, resampled_labels):
        cropped_image, cropped_label = crop_volume(image, label)
        cropped_images.append(cropped_image)
        cropped_labels.append(cropped_label)

    average_after_reg = compute_average_volume(resampled_images)
    average_after_crop = compute_average_volume(cropped_images)
    output_filename_after = os.path.join(output_dir, f"{reference_image_filename}_average_after.mha")
    output_filename_before = os.path.join(output_dir, f"{reference_image_filename}_average_before.mha")
    output_filename_after_crop = os.path.join(output_dir, f"{reference_image_filename}_average_after_crop.mha")
    sitk.WriteImage(average_after_reg, output_filename_after)
    sitk.WriteImage(average_before_reg, output_filename_before)
    sitk.WriteImage(average_after_crop, output_filename_after_crop)

    input_image = images[reference_index]
    label_image = labels[reference_index]

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
    print(reference_image_filename, input_image.GetSize(), input_image.GetSpacing(), input_image.GetOrigin(),
          input_image.GetDirection(), min_greyval, max_greyval)
    
    # img filename: ct_train_1001_image_isotropic.mha
    # image size: (192, 192, 192)
    # image spacing: (1.0, 1.0, 1.0)
    # image origin: (0.0, 0.0, 0.0)
    # image orientation: (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) = 3x3 matrix
    # image min greyvalue: -1023 
    # image max greyvalue: 3001


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
