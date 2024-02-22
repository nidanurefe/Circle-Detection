import cv2
import math

def find_circle_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filtered_frame = cv2.bilateralFilter(gray, 5, 175, 175)
    edge_detected_frame = cv2.Canny(filtered_frame, 100, 200)
    # The Canny edge detection algorithm aims to identify edges in 
    # an image accurately with minimal false positives or false negatives.

    contours, _ = cv2.findContours(edge_detected_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # mode: Specifies the contour retrieval mode. This parameter can take different values such as cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_TREE, etc. cv2.RETR_EXTERNAL is used to find the outer contours, 
    # while cv2.RETR_TREE is used to find hierarchical contours.

    # method: Specifies the contour approximation method. This parameter determines the contour approximation or simplification methods. It usually takes values like cv2.CHAIN_APPROX_SIMPLE or cv2.CHAIN_APPROX_NONE. 
    # cv2.CHAIN_APPROX_SIMPLE simplifies the contours, while cv2.CHAIN_APPROX_NONE does not simplify the contours.

    contour_list = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
        # This function approximates a contour with a polygon,
        # True is the flag indicating whether the contour is closed. In a closed contour, the start and end points are connected.
        # The expression epsilon = 0.01 * cv2.arcLength(contour, True) sets the epsilon value to 1% of the original contour's length. 
        # This determines how close the approximate polygon will be to the original contour.
        area = cv2.contourArea(contour)
        if len(approx) > 5 and area > 0.5:
            contour_list.append(contour)
        cv2.drawContours(frame, contour_list, -1, (0, 255, 0), 2)

    for i in contour_list:
        M = cv2.moments(i)
        # Moments are a set of statistical measures that describe the distribution of intensity or color in an image
        #'m00': Area of the contour.
        #'m10', 'm01': First-order moments of the contour along the x and y axes, respectively.
        if M['m00'] != 0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (x - 20, y - 20), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 1)
            z = 12  #this represents z coordinate of the circle center, this will be calculated using sensors.
            return x, y, z

    return None

def detect_circles(video_path):
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        print("Error: Unable to open video.")
        return

    while True:
        ret, frame = vid.read()
        # Ret is a boolean value indicating whether the call to read() was successful. 
        circle_center = find_circle_center(frame)
        if circle_center:
            roll, pitch, yaw = calculate_movements(circle_center, frame)
            print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

        cv2.imshow('Circle Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

#roll : rotation about x axis
#pitch : rotation about y axis
#yaw : rotation about z axis

def calculate_movements(circle_center, frame):
    camera_position = (frame.shape[1] // 2, frame.shape[0] // 2, 0)
    vector3D = (circle_center[0] - camera_position[0], 
                circle_center[1] - camera_position[1],
                circle_center[2] - camera_position[2])

    yaw = math.atan2(vector3D[1], vector3D[0])
    pitch = math.atan2(vector3D[2], math.sqrt(vector3D[2]**2 + vector3D[1]**2 + vector3D[0]**2))
    roll = 0 # Assuming no roll is needed to align with the center from the side

    return roll, pitch, yaw





if __name__ == "__main__":
    video_path = input("Enter video path:")
    detect_circles(video_path)
