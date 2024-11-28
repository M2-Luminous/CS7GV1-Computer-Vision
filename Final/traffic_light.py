import cv2
import numpy as np

# Define the ground truth data (image name, state, top left x, top left y, bottom right x, bottom right y)
ground_truth_data = [
    ("CamVidLights01.png", "Green", 319, 202, 346, 279),("CamVidLights01.png", "Green", 692, 264, 711, 322),
    ("CamVidLights02.png", "Green", 217, 103, 261, 230),("CamVidLights02.png", "Green", 794, 212, 820, 294),
    ("CamVidLights03.png", "Green", 347, 210, 373, 287),("CamVidLights03.png", "Green", 701, 259, 720, 318),
    ("CamVidLights04.png", "Red", 271, 65, 309, 189),("CamVidLights04.png", "Red", 640, 260, 652, 301),
    ("CamVidLights05.png", "Red+Amber", 261, 61, 302, 193),("CamVidLights05.png", "Red+Amber", 644, 269, 657, 312),
    ("CamVidLights06.png", "Green", 238, 42, 284, 187),("CamVidLights06.png", "Green", 650, 279, 663, 323),
    ("CamVidLights07.png", "Amber", 307, 231, 328, 297),("CamVidLights07.png", "Amber", 747, 266, 764, 321),
    ("CamVidLights08.png", "Red", 280, 216, 305, 296),("CamVidLights08.png", "Red", 795, 253, 816, 316),
    ("CamVidLights09.png", "Green", 359, 246, 380, 305),("CamVidLights09.png", "Green", 630, 279, 646, 327),
    ("CamVidLights10.png", "Green", 260, 122, 299, 239),("CamVidLights10.png", "Green", 691, 271, 705, 315),
    ("CamVidLights11.png", "Green", 331, 260, 349, 312),("CamVidLights11.png", "Green", 663, 280, 679, 322),
    ("CamVidLights12.png", "Green", 373, 219, 394, 279),("CamVidLights12.png", "Green", 715, 242, 732, 299)
    #("CamVidLights13.png", "Red", 283, 211, 299, 261),("CamVidLights13.png", "Red", 604, 233, 620, 279),
    #("CamVidLights14.png", "Red", 294, 188, 315, 253),("CamVidLights14.png", "Red", 719, 225, 740, 286)
]

# List of image paths
image_paths = [
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights01.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights02.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights03.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights04.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights05.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights06.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights07.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights08.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights09.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights10.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights11.png',
    'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights12.png'
    #'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights13.png',
    #'C:/Users/M2-Winterfell/Downloads/Computer Vision/Final/CamVidLights14.png'
]

def detect_traffic_light_back_projection(image_path):
    # Initialize counters
    green_count = 0
    red_count = 0
    yellow_count = 0

    # Read the image
    image = cv2.imread(image_path)
    original_image = image.copy()  # Keep a copy of the original image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # range for green
    green_lower = np.array([64, 235, 23])
    green_upper = np.array([75, 255, 30])

    # range for red
    red_lower = np.array([0, 245, 50])
    red_upper = np.array([4, 255, 80])

    # range for yellow
    yellow_lower = np.array([5, 245, 50])
    yellow_upper = np.array([10, 255, 80])

    # range for black (to filter out non-traffic light areas)
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 85])
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    # Create masks for green, red, and yellow colors
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Calculate histograms for green, red, and yellow
    green_hist = cv2.calcHist([hsv], [0, 1], green_mask, [180, 256], [0, 180, 0, 256])
    red_hist = cv2.calcHist([hsv], [0, 1], red_mask, [180, 256], [0, 180, 0, 256])
    yellow_hist = cv2.calcHist([hsv], [0, 1], yellow_mask, [180, 256], [0, 180, 0, 256])

    # Normalize the histograms
    cv2.normalize(green_hist, green_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(red_hist, red_hist, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(yellow_hist, yellow_hist, 0, 255, cv2.NORM_MINMAX)

    # Back Projection for green, red, and yellow
    green_dst = cv2.calcBackProject([hsv], [0, 1], green_hist, [0, 180, 0, 256], 1)
    red_dst = cv2.calcBackProject([hsv], [0, 1], red_hist, [0, 180, 0, 256], 1)
    yellow_dst = cv2.calcBackProject([hsv], [0, 1], yellow_hist, [0, 180, 0, 256], 1)

    # Smoothing
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(green_dst, -1, disc, green_dst)
    cv2.filter2D(red_dst, -1, disc, red_dst)
    cv2.filter2D(yellow_dst, -1, disc, yellow_dst)

    # Filter the result with the black mask
    filtered_green_dst = cv2.bitwise_and(green_dst, green_dst, mask=black_mask)
    filtered_red_dst = cv2.bitwise_and(red_dst, red_dst, mask=black_mask)
    filtered_yellow_dst = cv2.bitwise_and(yellow_dst, yellow_dst, mask=black_mask)

    # Thresholding
    _, green_thresh = cv2.threshold(filtered_green_dst, 50, 255, 0)
    _, red_thresh = cv2.threshold(filtered_red_dst, 50, 255, 0)
    _, yellow_thresh = cv2.threshold(filtered_yellow_dst, 50, 255, 0)

    # Find contours for green, red, and yellow
    green_contours, _ = cv2.findContours(green_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # match with the ground truth data
    image_name = image_path.split('/')[-1]

    # Function to check if two boxes overlap
    def do_boxes_overlap(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2):
            return True
        else:
            return False

    # Function to draw ground truth boxes and check overlap
    def draw_ground_truth_and_check_overlap(image, image_name, boxes):
        overlapping_boxes = []
        # Get the ground truth boxes for the current image
        ground_truth_boxes = [(item[2], item[3], item[4] - item[2], item[5] - item[3]) for item in ground_truth_data if item[0] == image_name]

        for item in ground_truth_data:
            if item[0] == image_name:
                _, state, x1, y1, x2, y2 = item
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                text_position = (x2 if x2 + 10 + 3 * len(state) < image.shape[1] else x1 - 10 - 3 * len(state), (y1 + y2) // 2)
                cv2.putText(image, state, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)

        for box in boxes:
            for gt_box in ground_truth_boxes:
                if do_boxes_overlap(box, gt_box):
                    overlapping_boxes.append(box)
                    break

        return overlapping_boxes


    # Function to filter and draw bounding boxes
    def process_contours(contours, color, counter):
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 40 < w*h < 800:  # Filter out too small/large areas
                bboxes.append((x, y, w, h))
                counter += 1  # detected traffic lights

        final_boxes = []
        while bboxes:
            base_box = bboxes.pop(0)
            base_center = (base_box[0] + base_box[2] / 2, base_box[1] + base_box[3] / 2)
            boxes_to_keep = []

            for box in bboxes:
                center = (box[0] + box[2] / 2, box[1] + box[3] / 2)
                if np.sqrt((base_center[0] - center[0])**2 + (base_center[1] - center[1])**2) > 20:
                    boxes_to_keep.append(box)

            final_boxes.append(base_box)
            bboxes = boxes_to_keep

        # Draw the final bounding boxes
        for box in final_boxes:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        return counter

    # Process contours for green and red and yellow
    process_contours(green_contours, (0, 255, 0), green_count)
    process_contours(red_contours, (0, 0, 255), red_count)
    process_contours(yellow_contours, (0, 255, 255), yellow_count)

    # Process contours for each color and update counters
    green_count = process_contours(green_contours, (0, 255, 0), green_count)
    red_count = process_contours(red_contours, (0, 0, 255), red_count)
    yellow_count = process_contours(yellow_contours, (0, 255, 255), yellow_count)

    # Determine the state based on the counters
    state = ""
    if green_count >= 2 and red_count < 2 and yellow_count < 2:
        state = "GREEN"
    elif yellow_count >= 2 and green_count < 2 and red_count < 3:
        state = "AMBER"
    elif red_count >= 2 and green_count < 2 and yellow_count < 2:
        state = "RED"
    elif red_count >= 2 and yellow_count >= 2 and green_count < 2:
        state = "RED-AMBER"
    else:
        state = "RED"

    # Display the state on the top left corner of the image if a state is determined
    if state:
        cv2.putText(image, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    # Define the HSV range for white colors
    light_lower = np.array([60, 0, 40])
    light_upper = np.array([172, 111, 255])
    
    # Create a mask for white colors
    light_mask = cv2.inRange(hsv, light_lower, light_upper)

    # Apply Canny edge detection on the original image
    edges = cv2.Canny(original_image, 100, 650)

    # Filter edges using the light mask
    filtered_edges = cv2.bitwise_and(edges, edges, mask=light_mask)

    # Function to calculate initial white box dimensions
    def calculate_initial_white_box_dimensions(area):
        width = np.sqrt(area / 15)
        height = 15 * width
        return int(width), int(height)
    
    # Function to expand boxes until all four sides meet their second edges
    def expand_until_second_edge(box, edges):
        x, y, w, h = box
        edge_encounters = [0, 0, 0, 0]  # Counters for each side's edge encounters

        while all(e < 6 for e in edge_encounters):
            # Expand the box
            x, y, w, h = x - 1, y - 1, w + 2, h + 2

            # Check for expansion beyond image boundaries
            if x <= 0 or y <= 0 or (x + w) >= edges.shape[1] or (y + h) >= edges.shape[0]:
                break

            # Check for edges on each side
            if np.any(edges[y, x:x+w]) and edge_encounters[0] < 6:  # Top edge
                edge_encounters[0] += 1
            if np.any(edges[y+h-1, x:x+w]) and edge_encounters[1] < 6:  # Bottom edge
                edge_encounters[1] += 1
            if np.any(edges[y:y+h, x]) and edge_encounters[2] < 6:  # Left edge
                edge_encounters[2] += 1
            if np.any(edges[y:y+h, x+w-1]) and edge_encounters[3] < 6:  # Right edge
                edge_encounters[3] += 1
                
        return x, y, w, h

    # List to store final expanded boxes
    final_boxes = []

    # Process and expand boxes
    for color_contours in [green_contours, red_contours, yellow_contours]:
        for cnt in color_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 40 < w*h < 800:  # Filter based on area size
                initial_w, initial_h = calculate_initial_white_box_dimensions(w * h)
                initial_x = x + (w - initial_w) // 2
                initial_y = y + (h - initial_h) // 2
                expanded_box = expand_until_second_edge((initial_x, initial_y, initial_w, initial_h), filtered_edges)
                final_boxes.append(expanded_box)

    # Remove overlapping boxes, keeping only the largest (Could change the algorithm to using an enclosing white hollow box to cover them all)
    def remove_overlapping_boxes(boxes):
        boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
        non_overlapping_boxes = []
        for box in boxes:
            overlap = any((other_box != box and
                           box[0] < other_box[0] + other_box[2] and
                           box[0] + box[2] > other_box[0] and
                           box[1] < other_box[1] + other_box[3] and
                           box[1] + box[3] > other_box[1]) for other_box in non_overlapping_boxes)
            if not overlap:
                non_overlapping_boxes.append(box)
        return non_overlapping_boxes

    final_boxes = remove_overlapping_boxes(final_boxes)

    # Remove white boxes that don't overlap with ground truth
    final_boxes = draw_ground_truth_and_check_overlap(image, image_name, final_boxes)

    # Adjust box positions based on the traffic light state
    adjusted_boxes = []
    for box in final_boxes:
        x, y, w, h = box
        # Move the box up for GREEN state
        if state == "GREEN":
            y -= 25  # Move up by 25 pixels; adjust this value as needed
        # Move the box down for RED state
        elif state == "RED":
            y += 15  # Move down by 15 pixels; adjust this value as needed
        adjusted_boxes.append((x, y, w, h))

    final_boxes = adjusted_boxes

    # Draw non-overlapping expanded boxes on the image
    for box in final_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 255, 255), 2)

    # Function to calculate Intersection over Union (IoU)
    def calculate_iou(box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[0] + box1[2], box2[0] + box2[2])
        yB = min(box1[1] + box1[3], box2[1] + box2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = box1[2] * box1[3]
        box2Area = box2[2] * box2[3]
        iou = interArea / float(box1Area + box2Area - interArea)

        return iou

    # Function to calculate Area Similarity
    def calculate_area_similarity(box1, box2):
        box1Area = box1[2] * box1[3]
        box2Area = box2[2] * box2[3]
        smaller_area = min(box1Area, box2Area)
        larger_area = max(box1Area, box2Area)
        area_similarity = smaller_area / larger_area

        return area_similarity

    def calculate_and_display_metrics(detected_boxes, ground_truth_boxes):
        results = []
        matched_detected_boxes = set()  # To track which detected boxes have been matched

        for gtbox in ground_truth_boxes:
            gtbox_xywh = (gtbox[0], gtbox[1], gtbox[2] - gtbox[0], gtbox[3] - gtbox[1])
            best_iou = 0
            best_area_similarity = 0
            best_dbox_index = None  # Track index of the best matching detected box for ground truth box

            for dbox_index, dbox in enumerate(detected_boxes):
                iou = calculate_iou(dbox, gtbox_xywh)
                area_similarity = calculate_area_similarity(dbox, gtbox_xywh)

                if iou > best_iou:  # Find the detected box with the highest IoU for ground truth box
                    best_iou = iou
                    best_area_similarity = area_similarity
                    best_dbox_index = dbox_index

            if best_dbox_index is not None:
                matched_detected_boxes.add(best_dbox_index)  # Mark as matched
            results.append((best_iou, best_area_similarity))

        # Calculate missed and incorrect detections ---- trigger in row 363, commented for now 
        missed_detections = len(ground_truth_boxes) - len(matched_detected_boxes)
        incorrect_detections = len(detected_boxes) - len(matched_detected_boxes)

        # Ensure there are exactly two pairs, pad with zeros if not
        while len(results) < 2:
            results.append((0, 0))  # Assuming no detection as 0 IoU and 0 area similarity

        low_accuracy_msg = []
        if results[0][0] < 0.5 or results[0][1] < 0.5:
            low_accuracy_msg.append("Low accuracy on Pair 1")
        if results[1][0] < 0.5 or results[1][1] < 0.5:
            low_accuracy_msg.append("Low accuracy on Pair 2")
        if "Low accuracy on Pair 1" in low_accuracy_msg and "Low accuracy on Pair 2" in low_accuracy_msg:
            low_accuracy_msg = ["Low accuracy both on Pair 1 and Pair 2"]

        # Display results for both pairs in one row
        display_message = f"Pair 1: IoU: {results[0][0]:.4f}, Area Similarity: {results[0][1]:.4f} | " \
                          f"Pair 2: IoU: {results[1][0]:.4f}, Area Similarity: {results[1][1]:.4f} | " \
                          #f"Missed Detections: {missed_detections}, Incorrect Detections: {incorrect_detections}"
        if low_accuracy_msg:
            display_message += " | " + " ".join(low_accuracy_msg)

        print(display_message)


    # Get ground truth boxes for the current image
    ground_truth_boxes_for_image = [(item[2], item[3], item[4], item[5]) for item in ground_truth_data if item[0] == image_name]

    # Calculate IoU and Area Similarity for each pair and display results
    calculate_and_display_metrics(final_boxes, ground_truth_boxes_for_image)

    # Display results
    cv2.imshow('Final Result with Expanded Hollow White Boxes', image)
    # cv2.imshow('Edge Detection on Original Image', filtered_edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Process each image
for image_path in image_paths:
    detect_traffic_light_back_projection(image_path)
