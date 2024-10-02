import os
import argparse
import numpy as np
import cv2


def calculate_bounding_box(binary_image):
    """Calculate the bounding box (min_row, min_col, max_row, max_col) of the non-zero pixels in a binary image."""
    if np.any(binary_image) == False:
        return None
    rows = np.any(binary_image, axis=1)
    cols = np.any(binary_image, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    return (min_row, min_col, max_row, max_col)


def boxes_overlap(box1, box2):
    """Check if two bounding boxes overlap."""
    r1_min, c1_min, r1_max, c1_max = box1
    r2_min, c2_min, r2_max, c2_max = box2
    
    horizontal_overlap = (c1_min <= c2_max) and (c2_min <= c1_max)
    vertical_overlap = (r1_min <= r2_max) and (r2_min <= r1_max)
    
    return horizontal_overlap and vertical_overlap


def check_instance_overlap(instance1, instance2, track_results_dir, n_images):
    """
    Check if two instances overlap in the tracking results

    Inputs:
    instance1: instance ID of the first instance
    instance2: instance ID of the second instance
    track_results_dir: path to the folder containing the tracking results
    n_images: number of images in the tracking results
    """
    instance1_folder = os.path.join(track_results_dir, str(instance1))
    instance2_folder = os.path.join(track_results_dir, str(instance2))

    # check if the instance images exist (boolean array with shape (n_images,))
    instance1_exists = np.array([os.path.exists(os.path.join(instance1_folder, f'{i:05}.png')) for i in range(n_images)])
    instance2_exists = np.array([os.path.exists(os.path.join(instance2_folder, f'{i:05}.png')) for i in range(n_images)])

    # get the number of images where both instances exist
    both_exists = instance1_exists & instance2_exists
    n_images_both_exists = np.sum(both_exists)

    # Check if the bounding boxes overlap in any of the images
    overlap_count = 0
    for idx in np.where(both_exists)[0]:

        instance1_img = cv2.imread(os.path.join(instance1_folder, f'{idx:05}.png'), cv2.IMREAD_GRAYSCALE)
        instance2_img = cv2.imread(os.path.join(instance2_folder, f'{idx:05}.png'), cv2.IMREAD_GRAYSCALE)

        instance1_mask = instance1_img > 0
        instance2_mask = instance2_img > 0

        instance1_bbox = calculate_bounding_box(instance1_mask)
        instance2_bbox = calculate_bounding_box(instance2_mask)

        if instance1_bbox is None or instance2_bbox is None:
            continue

        if boxes_overlap(instance1_bbox, instance2_bbox):
            overlap_count += 1

    OVERLAP_THRESHOLD = 0.7
    if n_images_both_exists == 0:
        print(f'No overlapping images for instances {instance1} and {instance2}')
        return False
    elif overlap_count / n_images_both_exists >= OVERLAP_THRESHOLD:
        print(f'Overlap detected with with overlap ratio {overlap_count / n_images_both_exists}')
        return True
    else:
        print(f'No overlap detected with overlap ratio {overlap_count / n_images_both_exists}')
        return False
    

def merge_two_instances(instance1, instance2, new_instance_id, track_results_dir, n_images):
    """
    Merge two instances in the tracking results

    Inputs:
    instance1: instance ID of the first instance
    instance2: instance ID of the second instance
    new_instance_id: new instance ID after merging
    track_results_dir: path to the folder containing the tracking results
    n_images: number of images in the tracking results
    """
    instance1_folder = os.path.join(track_results_dir, str(instance1))
    instance2_folder = os.path.join(track_results_dir, str(instance2))
    new_instance_folder = os.path.join(track_results_dir, str(new_instance_id))

    if not os.path.exists(new_instance_folder):
        os.makedirs(new_instance_folder)

    for i in range(n_images):
        
        if os.path.exists(os.path.join(instance1_folder, f'{i:05}.png')) and os.path.exists(os.path.join(instance2_folder, f'{i:05}.png')):
            
            instance1_img = cv2.imread(os.path.join(instance1_folder, f'{i:05}.png'), cv2.IMREAD_GRAYSCALE)
            instance2_img = cv2.imread(os.path.join(instance2_folder, f'{i:05}.png'), cv2.IMREAD_GRAYSCALE)

            instance1_mask = instance1_img > 0
            instance2_mask = instance2_img > 0

            new_mask = np.logical_or(instance1_mask, instance2_mask).astype(np.uint8) * 255
            new_mask = cv2.merge([new_mask, new_mask, new_mask])

            cv2.imwrite(os.path.join(new_instance_folder, f'{i:05}.png'), new_mask)

        elif os.path.exists(os.path.join(instance1_folder, f'{i:05}.png')):
            os.system(f'cp {os.path.join(instance1_folder, f"{i:05}.png")} {os.path.join(new_instance_folder, f"{i:05}.png")}')

        elif os.path.exists(os.path.join(instance2_folder, f'{i:05}.png')):
            os.system(f'cp {os.path.join(instance2_folder, f"{i:05}.png")} {os.path.join(new_instance_folder, f"{i:05}.png")}')


def merge_instances(track_results_dir):
    """
    Merge the instances of the tracking results

    Inputs:
    track_results_dir: path to the folder containing the tracking results
    """
    vis_folder = os.path.join(track_results_dir, 'Visualizations')
    n_images = len(os.listdir(vis_folder))

    # get instance IDs
    instance_ids = []
    for folder_name in os.listdir(track_results_dir):
        if folder_name.isdigit():
            instance_ids.append(int(folder_name))
            print(f'Instance ID: {folder_name}')

    # iteratively merge instances if their bounding boxes overlapped
    changed = True
    while changed:
        changed = False
        to_add = []
        to_remove = []
        
        keys = list(instance_ids)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[i] not in to_remove and keys[j] not in to_remove:
                    if check_instance_overlap(keys[i], keys[j], track_results_dir, n_images):
                        print(f'Merging instances {keys[i]} and {keys[j]}')
                        new_instance_id = keys[i] + keys[j]
                        to_add.append(new_instance_id)
                        to_remove.append(keys[i])
                        to_remove.append(keys[j])
                        merge_two_instances(keys[i], keys[j], new_instance_id, track_results_dir, n_images)
                        changed = True

        for key in to_remove:
            instance_ids.remove(key)
            os.system(f'rm -r {os.path.join(track_results_dir, str(key))}')   # remove the instance folder

        instance_ids.extend(to_add)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--track_results_dir', type=str, required=True,
                        help='Path to the folder containing the tracking results')
    args = parser.parse_args()

    merge_instances(args.track_results_dir)