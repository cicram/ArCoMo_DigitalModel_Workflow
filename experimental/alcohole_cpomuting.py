import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_bac(beers_consumed, time_hours, weight_kg, gender='male'):
    # Constants for the Widmark formula
    r = 0.55  # Widmark factor for males
    if gender.lower() == 'female':
        r = 0.68  # Widmark factor for females

    body_water_constant = 0.58 if gender.lower() == 'male' else 0.49

    # Standard drinks contain approximately 14 grams of pure alcohol
    grams_of_alcohol = beers_consumed * 14

    # Calculate blood alcohol content (BAC) using the Widmark formula
    bac = (grams_of_alcohol / (body_water_constant * weight_kg)) * 100

    # Consider the reduction of BAC over time (0.1 promille per hour)
    decay = 0.1 * time_hours
    bac -= decay

    return bac

def plot_bac_over_time(beers_consumed, total_time_hours, weight_kg, gender='male'):
    # Generate data for the 3D plot
    time_points = np.linspace(0, total_time_hours, 100)
    beers_points = np.linspace(0, beers_consumed, 100)
    time_mesh, beers_mesh = np.meshgrid(time_points, beers_points)
    bac_values = calculate_bac(beers_mesh, time_mesh, weight_kg, gender)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(beers_mesh, time_mesh, bac_values, cmap='viridis')

    # Set labels
    ax.set_xlabel('Beers Consumed')
    ax.set_ylabel('Time (hours)')
    ax.set_zlabel('Blood Alcohol Content (%)')

    # Display the plot
    plt.show()

# Example usage:
plot_bac_over_time(beers_consumed=10, total_time_hours=5, weight_kg=70, gender='male')
