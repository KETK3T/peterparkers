import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import matplotlib.ticker as ticker

import icm20x_icm20948_simpletest as ist

imu = ist.icm

#x=time,y1=x-direction,y2=y-direction

# Create figure and axes
#fig, ax = plt.subplots()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4)) # 1 row, 3 columns
#3 graphs side by side
x_data, y1_data, y2_data = [], [], []
line1, = axes[0].plot(x_data, y1_data, label='X Direction') 
line2, = axes[0].plot(x_data, y2_data, label='Y Direction') 
line3, = axes[1].plot(x_data, y2_data, label='Y Direction') 
line4, = axes[1].plot(x_data, y1_data, label='X Direction')
line5, = axes[2].plot(x_data, y2_data, label='Y Direction')
line6, = axes[2].plot(x_data, y2_data, label='Y Direction')


start_time = time.time() #seconds
window_length = 30 #seconds to display

# Function to update the plot
def update(frame):

    # Simulate new data acquisition
    new_x = time.time() - start_time # Use current time for x-axis

    '''
    new_y1 = np.random.randint(-10,10) # Generate random y-value; ax
    new_y2 = np.random.randint(-10,10) #ay
    new_y1v = np.random.randint(-10,10) #vx
    new_y2v = np.random.randint(-10,10) #vy
    new_y1s = np.random.randint(-10,10) #sx
    new_y2s = np.random.randint(-10,10) #sy
    '''

    #ax,ay = icm.acceleration[:2]
    #before values added to graph, they need to be filtered for noise
    #probably will need filter right here
    new_y1 = imu.acceleration[0]
    new_y2 = imu.acceleration[1]

    x_data.append(new_x)
    y1_data.append(new_y1)
    y2_data.append(new_y2)

    # Keep only the last 100 data points for a scrolling effect
    #if len(x_data) > 10:
    while x_data and (new_x - x_data[0]) > window_length: # Remove old data outside the 10-second window
        x_data.pop(0)
        y1_data.pop(0)
        y2_data.pop(0)

    #Update line data
    line1.set_data(x_data, y1_data)
    line2.set_data(x_data, y2_data)
    
    #Rescale axes
    axes[0].relim() # Recalculate limits
    axes[0].autoscale_view(scalex=False) # Autoscale axes
    # Set x-axis to show only last 10 seconds
    axes[0].set_xlim(new_x - window_length, new_x)

    #Add labels and styling
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Acceleration")
    axes[0].set_title("Acceleration in the X and Y Directions")
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].legend(loc='upper left')

    # Force x-axis ticks at 1-second intervals
    axes[0].xaxis.set_major_locator(ticker.MultipleLocator(5))
    axes[0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    return line1, line2, line3, line4, line5, line6

# Create the animation
ani = animation.FuncAnimation(fig, update, interval=100) # Update every 100ms

plt.show()



