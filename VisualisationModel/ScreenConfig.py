from VisualisationModel.Color import Color


class ScreenConfig:
    screen_width = 700
    screen_height = 700
    background_color = Color.white
    bounding_box = ((-10, -3), (10, 3))
    viewport_center = ((max(bounding_box[0][0], bounding_box[1][0]) + min(bounding_box[0][0], bounding_box[1][0])) // 2,
                       ((max(bounding_box[0][1], bounding_box[1][1]) + min(bounding_box[0][1], bounding_box[1][1]))) // 2)
    y_pixels_for_one = screen_height / (max(bounding_box[0][1], bounding_box[1][1]) - min(bounding_box[0][1], bounding_box[1][1]))
    x_pixels_for_one = screen_width / (max(bounding_box[0][0], bounding_box[1][0]) - min(bounding_box[0][0], bounding_box[1][0]))
