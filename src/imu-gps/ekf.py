import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
import csv

class EKF(ExtendedKalmanFilter):
    def __init__(self):
        super().__init__(dim_x=5, dim_z=2)

        # State vector
        # Contains x, y, vx, vy, and theta
        # x is position in x direction
        # y is position in y direction
        # vx is velocity in x direction
        # vy is velocity in y direction
        # theta is heading orientation
        self.x = np.zeros(5)

        # Covariance (P matrix)
        self.P = np.eye(5) * 1.0

        # Process noise (Q matrix), this will need to be adjusted to the IMU datasheets
        self.Q = np.diag([0.1, 0.1, 0.5, 0.5, 0.05])

        # Measurement noise (GPS) (R matrix), this will need to be adjusted to the GPS datasheets
        self.R = np.diag([3.0, 3.0])

        # For creating test data -> used in load function
        with open("load-data/output.csv", "w", newline="", encoding="utf-8") as f:
            self.writer = csv.writer(f)
        
        self.writer.writerow(["id", "latitude", "longitude", "x-velocity", "y-velocity", "heading"])
        self.count = 1 # csv row id

    # Prediction step
    # dt is change in time, will use timestamps to calculate
    def predict_f(self, x, dt, ax, ay, gyro_z):
        x_pos, y_pos, vx, vy, theta = x

        # Rotate acceleration into world frame
        # Changes coordinates for acceleration from vehicle coordinates to map coordinates
        # Better corresponds with GPS and state vector
        a_wx = ax*np.cos(theta) - ay*np.sin(theta)
        a_wy = ax*np.sin(theta) + ay*np.cos(theta)

        # Predicts using kinematics
        x_pos += vx*dt + 0.5*a_wx*dt*dt
        y_pos += vy*dt + 0.5*a_wy*dt*dt
        vx    += a_wx*dt
        vy    += a_wy*dt
        theta += gyro_z*dt

        # Returns updated state vector
        return np.array([x_pos, y_pos, vx, vy, theta])

    # F matrix is Jacobian of motion model
    def F_jacobian(self, x, dt, ax, ay):
        _, _, vx, vy, theta = x

        F = np.eye(5)
        F[0,2] = dt         # dx/dvx
        F[1,3] = dt         # dy/dvy

        # Derivatives wrt theta (due to acceleration rotation)
        F[0,4] = (-ax*np.sin(theta) - ay*np.cos(theta)) * dt
        F[1,4] = ( ax*np.cos(theta) - ay*np.sin(theta)) * dt

        return F

    # Measurement function h(x)
    def H(self, x):
        # GPS measures only position
        return np.array([x[0], x[1]])

    # H matrix = Jacobian of measurement model
    def H_jacobian(self, x):
        H = np.zeros((2, 5))
        H[0,0] = 1
        H[1,1] = 1
        return H

    def __str__(self):
        return (
            f"Position: Lat:{self.x[0]:.3f}, Long:{self.x[1]:.3f}\n"
            f"Velocity: X:{self.x[2]:.3f}, Y:{self.x[3]:.3f} m/s\n"
            f"Heading: {self.x[4]:.3f} radians\n"
        )
    
    def load(self):
        self.writer.writerow([self.count,self.x[0],self.x[1],self.x[2],self.x[3],self.x[4]])
        self.count += 1

    def getstate(self):
        return self.x
    
    
