import logging
import math
import cv2
import numpy as np

# ============================================================================
CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]
# ============================================================================

class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.counted1 = False
        self.counted2 = False
        self.counted3 = False
        self.counted4 = False
        # self.counted4 = False
        # self.counted5 = False
        # self.counted6 = False

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0

    def draw(self, output_image):
        car_colour = CAR_COLOURS[self.id % len(CAR_COLOURS)]
        for point in self.positions:
            ls = [ x[1] for x in self.positions ]
            cv2.circle(output_image, point[1], 2, car_colour, -1)
            cv2.polylines(output_image, [np.int32(ls)]
                , False, car_colour, 1)


# ============================================================================== #
    # Divider Equation function to check if vehicle has crossed the divider
# ============================================================================== #
def Divider_eqn(pt1,pt2,vehicle):
    slope = (pt2[1] - pt1[1])/(pt2[0] - pt1[0]) # calculating the slope
    c = pt2[1]-slope*pt2[0] # calculating the constant
    pos_last = vehicle.positions[-1][1] # latest position
    pos_othr = vehicle.positions[-4][1] # some previous position
    

    sign1 = pos_last[1]-slope*pos_last[0] - c # value when last point is substituted in line
    sign2 = pos_othr[1]-slope*pos_othr[0] - c # value when other point is substituted in line

    # checking if both the points lie on the opposite side of the divider
    if(sign1*sign2 < 0):
        if(pos_last[0] > pos_othr[0]):
            return 1 # towards right
        else:
            return -1 # towards left
    # not crossed the divider
    return 0
# =============================================================================== #
                            # Vehicle counting class
# =============================================================================== #
class VehicleCounter(object):
    def __init__(self, shape, divider1, divider2, divider3, divider4):

        self.divider1a_x, self.divider1a_y = divider1[0][0], divider1[0][1]
        self.divider1b_x, self.divider1b_y = divider1[1][0], divider1[1][1]
        self.divider2a_x, self.divider2a_y = divider2[0][0], divider2[0][1]
        self.divider2b_x, self.divider2b_y = divider2[1][0], divider2[1][1]
        self.divider3a_x, self.divider3a_y = divider3[0][0], divider3[0][1]
        self.divider3b_x, self.divider3b_y = divider3[1][0], divider3[1][1]
        self.divider3b_x, self.divider3b_y = divider3[1][0], divider3[1][1]
        self.divider4a_x, self.divider4a_y = divider4[0][0], divider4[0][1]
        self.divider4b_x, self.divider4b_y = divider4[1][0], divider4[1][1]
        
        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count1 = 0
        self.vehicle_count1_up = 0
        self.vehicle_count1_down = 0
        self.vehicle_count2_up = 0
        self.vehicle_count2_down = 0
        self.vehicle_count3_up = 0
        self.vehicle_count3_down = 0
        self.vehicle_count4_left = 0
        self.vehicle_count4_right = 0
        self.max_unseen_frames = 6

    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.
        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[1][0] - a[1][0])
        dy = float(b[1][1] - a[1][1])

        distance = math.sqrt(dx**2 + dy**2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0        

        return distance, angle 


    @staticmethod
    def is_valid_vector(a):
        distance, angle = a
        threshold_distance = 45
        return distance <= threshold_distance

    def update_vehicle(self, vehicle, matches):
        # Find if any of the matches fits this vehicle
        for i, match in enumerate(matches):
            centroid = match
            vector = self.get_vector(vehicle.positions[-1], centroid)
            if self.is_valid_vector(vector):
                vehicle.add_position(centroid)
                return i
        # No matches fit...        
        vehicle.frames_since_seen += 1
        return None


    def update_count(self, matches, output_image = None):
        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)

        my_div4x = (self.divider4a_x + self.divider4b_x)/2
        my_div3y = (self.divider3a_y + self.divider3b_y)/2
        my_div2y = (self.divider2a_y + self.divider2b_y)/2
        my_div1y = (self.divider1a_y + self.divider1b_y)/2

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            # For divider 3 MIDDLE ONE
            if not vehicle.counted3 and len(vehicle.positions) > 6 and (self.divider3b_x < vehicle.positions[-1][1][0] < self.divider3a_x):
                if((vehicle.positions[-1][1][1] >= my_div3y >= vehicle.positions[-5][1][1])):
                    self.vehicle_count3_down += 1
                    vehicle.counted3 = True
                elif(vehicle.positions[-1][1][1] <= my_div3y <= vehicle.positions[-5][1][1]):
                    self.vehicle_count3_up += 1
                    vehicle.counted3 = True
            # For divider 2 LEFT ONE
            if not vehicle.counted2 and len(vehicle.positions) > 6 and (self.divider2b_x < vehicle.positions[-1][1][0] < self.divider2a_x):
                if((vehicle.positions[-1][1][1] >= my_div2y >= vehicle.positions[-5][1][1])):
                    self.vehicle_count2_down += 1
                    vehicle.counted2 = True
                elif(vehicle.positions[-1][1][1] <= my_div2y <= vehicle.positions[-5][1][1]):
                    self.vehicle_count2_up += 1
                    vehicle.counted2 = True
            # For divider 1 RIGHT ONE
            if not vehicle.counted1 and len(vehicle.positions) > 6 and (self.divider1b_x < vehicle.positions[-1][1][0] < self.divider1a_x):
                if((vehicle.positions[-1][1][1] >= my_div1y >= vehicle.positions[-5][1][1])):
                    self.vehicle_count1_down += 1
                    vehicle.counted1 = True
                elif(vehicle.positions[-1][1][1] <= my_div1y <= vehicle.positions[-5][1][1]):
                    self.vehicle_count1_up += 1
                    vehicle.counted1 = True
            # For divider 4 ROUND ABOUT
            if not vehicle.counted4 and len(vehicle.positions) > 5 and (self.divider4a_y < vehicle.positions[-1][1][1] < self.divider4b_y):
                res = Divider_eqn((self.divider4b_x, self.divider4b_y),(self.divider4a_x, self.divider4a_y),vehicle)
                if(res):
                    if(res>0):
                        print("up right here")
                        self.vehicle_count4_right += 1
                        vehicle.counted4 = True
                    else:
                        self.vehicle_count4_left += 1
                        print("down right here")
                        vehicle.counted4 = True   

        # Optionally draw the vehicles on an image
        if output_image is not None:
            for vehicle in self.vehicles:
                vehicle.draw(output_image)

            # For divider 2 MIDDLE ONE
            cv2.putText(output_image, ("leftDown:%02d" % self.vehicle_count2_down), (512, 100)
                , cv2.FONT_HERSHEY_PLAIN, 1.7, (55, 55, 255), 2)
            cv2.putText(output_image, ("leftUp:%02d" % self.vehicle_count2_up), (512, 150)
                , cv2.FONT_HERSHEY_PLAIN, 1.7, (55, 55, 255), 2)
            # For divider 3 MIDDLE ONE
            cv2.putText(output_image, ("midDown:%02d" % self.vehicle_count3_down), (800, 180)
                , cv2.FONT_HERSHEY_PLAIN, 1.7, (127, 255, 255), 2)
            cv2.putText(output_image, ("midUp:%02d" % self.vehicle_count3_up), (800, 230)
                , cv2.FONT_HERSHEY_PLAIN, 1.7, (127, 255, 255), 2)
            # For divider 1 RIGHT ONE
            cv2.putText(output_image, ("rightDown:%02d" % self.vehicle_count1_down), (1080, 100)
            , cv2.FONT_HERSHEY_PLAIN, 1.7, (55, 255, 55), 2)
            cv2.putText(output_image, ("rightUp:%02d" % self.vehicle_count1_up), (1080, 150)
                , cv2.FONT_HERSHEY_PLAIN, 1.7, (55, 255, 55), 2)
            # For divider 4 ROUND ABOUT
            cv2.putText(output_image, ("round left:%02d" % self.vehicle_count4_left), (108, 100)
                , cv2.FONT_HERSHEY_PLAIN, 1.7, (55, 255, 55), 2)
            cv2.putText(output_image, ("round right:%02d" % self.vehicle_count4_right), (108, 150)
                , cv2.FONT_HERSHEY_PLAIN, 1.7, (55, 255, 55), 2)

        # Remove vehicles that have not been seen long enough
        removed = [ v.id for v in self.vehicles
            if v.frames_since_seen >= self.max_unseen_frames ]
        self.vehicles[:] = [ v for v in self.vehicles
            if not v.frames_since_seen >= self.max_unseen_frames ]