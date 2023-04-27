import cv2
import numpy as np

def risk_area(img, center, r, theta, alpha=0.4):
    # Define the angles for each of the three sectors
    angle1 = theta - 30
    angle2 = theta + 30
    overlay = img.copy()
    cv2.ellipse(overlay, center, (3*r, 3*r), 0, angle1, angle2, (0, 255, 255), -1)
    cv2.ellipse(overlay, center, (2*r, 2*r), 0, angle1, angle2, (0, 165, 255), -1)
    cv2.ellipse(overlay, center, (r, r), 0, angle1, angle2, (0, 0, 255), -1)
    
    # Apply transparency to the sectors using the cv2.addWeighted() function
    img = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)
    return img

if __name__ == "__main__":
    # Define the center of the sector and the radius r
    center = (200, 200)
    r = 100
    # Create a black image of the desired size
    img = 255*np.ones((800, 800, 3), dtype=np.uint8)

    img = risk_area(img, center, r, theta=30)

    # Display the resulting image
    cv2.imshow("Semitransparent Annular Sector", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()