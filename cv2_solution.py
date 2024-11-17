import numpy as np
import cv2
import typing
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import yaml


def get_matches(image1, image2):
    """
    Find matches between two images using SIFT and perform left-right consistency check.
    """
    sift = cv2.SIFT_create()
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    kp1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    kp2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches_1_to_2 = bf.knnMatch(descriptors1, descriptors2, k=2)
    matches_2_to_1 = bf.knnMatch(descriptors2, descriptors1, k=2)

    k_ratio = 0.75
    good_matches_1_to_2 = [
        m for m, n in matches_1_to_2 if m.distance < k_ratio * n.distance
    ]
    good_matches_2_to_1 = [
        m for m, n in matches_2_to_1 if m.distance < k_ratio * n.distance
    ]

    matches = []
    for match1 in good_matches_1_to_2:
        for match2 in good_matches_2_to_1:
            if match1.queryIdx == match2.trainIdx and match1.trainIdx == match2.queryIdx:
                matches.append(match1)
                break

    return kp1, kp2, matches


def get_second_camera_position(kp1, kp2, matches, camera_matrix):
    """
    Estimate the position and orientation of the second camera using essential matrix.
    """
    coordinates1 = np.array([kp1[match.queryIdx].pt for match in matches])
    coordinates2 = np.array([kp2[match.trainIdx].pt for match in matches])
    E, mask = cv2.findEssentialMat(coordinates1, coordinates2, camera_matrix)
    _, R, t, mask = cv2.recoverPose(E, coordinates1, coordinates2, camera_matrix)
    return R, t, E


def triangulation(camera_matrix, camera1_t, camera1_r, camera2_t, camera2_r, kp1, kp2, matches):
    """
    Perform triangulation to compute 3D points from matches.
    """
    camera1_projection = camera_matrix @ np.hstack((camera1_r, camera1_t))
    camera2_projection = camera_matrix @ np.hstack((camera2_r, camera2_t))

    points1 = np.array([kp1[match.queryIdx].pt for match in matches], dtype=np.float32).T
    points2 = np.array([kp2[match.trainIdx].pt for match in matches], dtype=np.float32).T

    points_4d = cv2.triangulatePoints(camera1_projection, camera2_projection, points1, points2)
    points_3d = (points_4d[:3] / points_4d[3]).T

    return points_3d


def resection(image1, image2, camera_matrix, matches, points_3d):
    """
    Compute the camera position of a new camera by solving the PnP problem.
    """
    kps_image1, kps_image2, refined_matches = get_matches(image1, image2)
    point_map = {match.queryIdx: points_3d[i] for i, match in enumerate(matches)}
    object_points = []
    image_points = []

    for m in refined_matches:
        if m.queryIdx in point_map:
            object_points.append(point_map[m.queryIdx])
            image_points.append(kps_image2[m.trainIdx].pt)

    object_points = np.array(object_points)
    image_points = np.array(image_points)

    _, rotation_vec, translation_vec, _ = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, np.zeros(5)
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)
    return rotation_matrix, translation_vec


def convert_to_world_frame(translation_vector, rotation_matrix):
    """
    Convert camera coordinates to world coordinates.
    """
    world_rotation_matrix = rotation_matrix.T
    world_position = -world_rotation_matrix @ translation_vector
    return world_position, world_rotation_matrix


def visualisation(camera_position1, camera_rotation1, camera_position2, camera_rotation2, camera_position3, camera_rotation3):
    """
    Visualize the positions and orientations of cameras in 3D space.
    """
    def plot_camera(ax, position, direction, label):
        color_scatter = 'blue' if label != 'Camera 3' else 'green'
        ax.scatter(position[0][0], position[1][0], position[2][0], color=color_scatter, s=100)
        color_quiver = 'red' if label != 'Camera 3' else 'magenta'

        ax.quiver(position[0][0], position[1][0], position[2][0], direction[0], direction[1], direction[2],
                  length=1, color=color_quiver, arrow_length_ratio=0.2)
        ax.text(position[0][0], position[1][0], position[2][0], label, color='black')

    camera_positions = [camera_position1, camera_position2, camera_position3]
    camera_directions = [camera_rotation1[:, 2], camera_rotation2[:, 2], camera_rotation3[:, 2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (position, direction) in enumerate(zip(camera_positions, camera_directions)):
        plot_camera(ax, position, direction, f'Camera {i + 1}')

    initial_elev, initial_azim = 0, 270
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.view_init(elev=initial_elev, azim=initial_azim)

    ax_elev_slider = plt.axes([0.1, 0.1, 0.65, 0.03])
    elev_slider = Slider(ax_elev_slider, 'Elev', 0, 360, valinit=initial_elev)

    ax_azim_slider = plt.axes([0.1, 0.05, 0.65, 0.03])
    azim_slider = Slider(ax_azim_slider, 'Azim', 0, 360, valinit=initial_azim)

    def update(val):
        ax.view_init(elev=elev_slider.val, azim=azim_slider.val)
        fig.canvas.draw_idle()

    elev_slider.on_changed(update)
    azim_slider.on_changed(update)

    plt.show()


def main():
    """
    Main entry point for the pipeline.
    """
    image1 = cv2.imread('./images/image0.jpg')
    image2 = cv2.imread('./images/image1.jpg')
    image3 = cv2.imread('./images/image2.jpg')

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    camera_matrix = np.array(config["camera_matrix"], dtype=np.float32, order='C')

    key_points1, key_points2, matches_1_to_2 = get_matches(image1, image2)
    R2, t2, _ = get_second_camera_position(key_points1, key_points2, matches_1_to_2, camera_matrix)

    triangulated_points = triangulation(
        camera_matrix,
        np.array([0, 0, 0]).reshape((3, 1)),
        np.eye(3),
        t2,
        R2,
        key_points1,
        key_points2,
        matches_1_to_2
    )

    R3, t3 = resection(image1, image3, camera_matrix, matches_1_to_2, triangulated_points)

    camera_position1, camera_rotation1 = convert_to_world_frame(np.array([0, 0, 0]).reshape((3, 1)), np.eye(3))
    camera_position2, camera_rotation2 = convert_to_world_frame(t2, R2)
    camera_position3, camera_rotation3 = convert_to_world_frame(t3, R3)

    visualisation(camera_position1, camera_rotation1, camera_position2, camera_rotation2, camera_position3, camera_rotation3)


if __name__ == "__main__":
    main()
