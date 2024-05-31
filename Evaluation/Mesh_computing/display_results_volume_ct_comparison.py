import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np


z_score_flag = 0
oct_section = 1


############################################
#ArCoMo1400
############################################

# Load the CSV file
df_gt = pd.read_csv('C:/Users/JL/Model_evaluation/phantom_CT/ArCoMo1400/volumes_result_gt.csv')
df_ct = pd.read_csv('C:/Users/JL/Model_evaluation/phantom_CT/ArCoMo1400/volumes_result_ct.csv')

fig = plt.figure()
plt.plot(df_gt['Centerline IDX'], df_gt['Volume'], marker='o', linestyle='-', color='black', label='Ground turth')
plt.plot(df_ct['Centerline IDX'], df_ct['Volume'], marker='o', linestyle='-', color='red', label='CT')
plt.xlabel('Centerline IDX')
plt.ylabel('Volume')
plt.title('Centerline IDX vs Volume')
plt.grid(True)
plt.legend()

ct_bifurcation = 51
gt_bifurcation = 167
shift = gt_bifurcation - ct_bifurcation

if oct_section:
    # Assuming the columns for volume measurements are named 'Volume'
    gt_volume = df_gt['Volume'][ct_bifurcation-30:ct_bifurcation+250]
    ct_volume = df_ct['Volume'][gt_bifurcation-30:gt_bifurcation+250]

else:
    gt_volume = df_gt['Volume']
    ct_volume = df_gt['Volume']

if z_score_flag:
    # Z-score normalization
    gt_volume = (gt_volume - gt_volume.mean()) / gt_volume.std()
    ct_volume = (ct_volume - ct_volume.mean()) / ct_volume.std()

colors = ['blue', 'green', 'red', 'purple', 'orange']
centerline_points = range(0,len(gt_volume))
# Plot 'Centerline IDX' vs 'Volume'
fig = plt.figure()
plt.plot(centerline_points, gt_volume, marker='o', linestyle='-', color='black', label='Ground turth')
plt.plot(centerline_points, ct_volume, marker='o', linestyle='-', color=colors[0], label='CT')
plt.xlabel('Centerline IDX')
plt.ylabel('Volume')
plt.title('Centerline IDX vs Volume')
plt.grid(True)
plt.legend()
plt.show()

if False:
    # Statistical Analysis
    statistical_data = {
        'Image correlation': volume_1,
        'Overlap': volume_2,
        'No correction': volume_3,
        'Pure CT': volume_4,
        'ICP': volume_5
    }

    analysis_data = []

    for method, areas in statistical_data.items():
        mean_abs_error = np.mean(np.abs(areas - gt_volume))
        median_abs_error = np.median(np.abs(areas - gt_volume))
        max_abs_error = np.max(np.abs(areas - gt_volume))

        mean_squared_error = np.mean((areas - gt_volume) ** 2)
        correlation_coefficient = np.corrcoef(areas, gt_volume)[0, 1]
        analysis_data.append([method, max_abs_error, mean_abs_error, median_abs_error, mean_squared_error, correlation_coefficient])

    # Create a DataFrame for the statistical analysis data
    analysis_df = pd.DataFrame(analysis_data, columns=['Method', 'Max Absolute Error', 'Median Absolute Error','Mean Absolute Error', 'Mean Squared Error', 'Correlation Coefficient'])

    # Save the DataFrame to an Excel file
    analysis_df.to_excel('Model_evaluation/AreaVolumeResults/Volume/statistical_analysis_ArCoMo300.xlsx', index=False)


    # Bland-Altman Plot
    def bland_altman_plot(data1, data2, title='', ax=None):
        if ax is None:
            ax = plt.gca()
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2
        ax.scatter(mean, diff, alpha=0.5)
        ax.axhline(np.mean(diff), color='black', linestyle='--')
        ax.set_xlabel('Mean of Measurements')
        ax.set_ylabel('Difference between Measurements')
        ax.set_title(title)
        ax.text(0.05, 0.95, f'Mean Diff: {np.mean(diff):.2f}\nSD: {np.std(diff):.2f}', 
                transform=ax.transAxes, verticalalignment='top')

    plt.figure(figsize=(8, 6))
    bland_altman_plot(gt_volume, volume_1, title='Bland-Altman Plot: Image correlation')
    plt.figure(figsize=(8, 6))
    bland_altman_plot(gt_volume, volume_2, title='Bland-Altman Plot: Overlap')
    plt.figure(figsize=(8, 6))
    bland_altman_plot(gt_volume, volume_3, title='Bland-Altman Plot: No correction')
    plt.figure(figsize=(8, 6))
    bland_altman_plot(gt_volume, volume_4, title='Bland-Altman Plot: Pure CT')
    plt.figure(figsize=(8, 6))
    bland_altman_plot(gt_volume, volume_5, title='Bland-Altman Plot: ICP')
    plt.show()


    # Fit lines between ground truth volumes and volumes of other methods
    slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = linregress(gt_volume, volume_1)
    slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = linregress(gt_volume, volume_2)
    slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = linregress(gt_volume, volume_3)
    slope_4, intercept_4, r_value_4, p_value_4, std_err_4 = linregress(gt_volume, volume_4)
    slope_5, intercept_5, r_value_5, p_value_5, std_err_5 = linregress(gt_volume, volume_5)

    # Calculate R-squared
    r_squared_1 = r_value_1 ** 2
    r_squared_2 = r_value_2 ** 2
    r_squared_3 = r_value_3 ** 2
    r_squared_4 = r_value_4 ** 2
    r_squared_5 = r_value_5 ** 2

    # Create a DataFrame for linear regression values
    data = {
        'Slope': [slope_1, slope_2, slope_3, slope_4, slope_5],
        'Intercept': [intercept_1, intercept_2, intercept_3, intercept_4, intercept_5],
        'R-value': [r_value_1, r_value_2, r_value_3, r_value_4, r_value_5],
        'P-value': [p_value_1, p_value_2, p_value_3, p_value_4, p_value_5],
        'Standard Error': [std_err_1, std_err_2, std_err_3, std_err_4, std_err_5],
        'R-value-squared': [r_squared_1, r_squared_2, r_squared_3, r_squared_4, r_squared_5]

    }

    df_linear_regression = pd.DataFrame(data, index=['Image correlation', 'Overlap', 'No correction', 'Pure CT', 'ICP'])

    # Save linear regression values to an Excel file
    df_linear_regression.to_excel('Model_evaluation/AreaVolumeResults/Volume/ArCoMo300_linear_regression_values.xlsx')


    # Plotting
    plt.figure(figsize=(10, 8))

    # Define colors for each pair of scatter plot and fitted line
    colors = ['blue', 'green', 'red', 'purple', 'orange']

    # Plotting data points and fitted lines with matching colors
    plt.plot(gt_volume, volume_1, 'o', color=colors[0], label='Image correlation')
    plt.plot(gt_volume, slope_1 * gt_volume + intercept_1, '--', color=colors[0], label=f'Fitted Line (slope={slope_1:.2f}, intercept={intercept_1:.2f})')

    plt.plot(gt_volume, volume_2, 'o', color=colors[1], label='Overlap')
    plt.plot(gt_volume, slope_2 * gt_volume + intercept_2, '--', color=colors[1], label=f'Fitted Line (slope={slope_2:.2f}, intercept={intercept_2:.2f})')

    plt.plot(gt_volume, volume_3, 'o', color=colors[2], label='No correction')
    plt.plot(gt_volume, slope_3 * gt_volume + intercept_3, '--', color=colors[2], label=f'Fitted Line (slope={slope_3:.2f}, intercept={intercept_3:.2f})')

    plt.plot(gt_volume, volume_4, 'o', color=colors[3], label='Pure CT')
    plt.plot(gt_volume, slope_4 * gt_volume + intercept_4, '--', color=colors[3], label=f'Fitted Line (slope={slope_4:.2f}, intercept={intercept_4:.2f})')

    plt.plot(gt_volume, volume_5, 'o', color=colors[4], label='ICP')
    plt.plot(gt_volume, slope_5 * gt_volume + intercept_5, '--', color=colors[4], label=f'Fitted Line (slope={slope_5:.2f}, intercept={intercept_5:.2f})')

    # Plotting the 45-degree line for reference
    min_val = min(min(gt_volume), min(volume_1), min(volume_2), min(volume_3), min(volume_4), min(volume_5))
    max_val = max(max(gt_volume), max(volume_1), max(volume_2), max(volume_3), max(volume_4), max(volume_5))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='45° Reference line')

    plt.xlabel('Ground Truth Volume')
    plt.ylabel('Measured Volume')
    plt.title('Measured Volume with Fitted Lines to Ground Truth')
    plt.legend()
    plt.show()

