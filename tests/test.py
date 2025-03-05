import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def find_peaks_in_dataframe(df, column_name, prominence=None, width=None, threshold=None):
    """
    Finds peaks in a specified column of a Pandas DataFrame.

    Args:
        df: The Pandas DataFrame.
        column_name: The name of the column to find peaks in.
        prominence (optional):  Minimum prominence of peaks.  See scipy.signal.find_peaks documentation.
        width (optional): Minimum width of peaks in samples. See scipy.signal.find_peaks documentation.
        threshold (optional): Minimum threshold for peaks. See scipy.signal.find_peaks documentation.

    Returns:
        A tuple containing:
            - A list of peak indices (integers).
            - A dictionary of properties for each peak (prominences, widths, etc.) as returned by scipy.signal.find_peaks.
            Returns None, None if the input is invalid or the column doesn't exist.
    """

    if not isinstance(df, pd.DataFrame) or df.empty or column_name not in df.columns:
        return None, None

    data = df[column_name].values

    peaks, properties = find_peaks(data, prominence=prominence, width=width, threshold=threshold)

    return peaks.tolist(), properties



# Example usage:
data = {'Value': [10, 12, 15, 13, 16, 18, 17, 20, 19, 22, 20, 18, 15, 17, 21, 23, 22, 20, 18, 15],
        'OtherColumn': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
df = pd.DataFrame(data)

peak_indices, peak_properties = find_peaks_in_dataframe(df, 'Value', prominence=2, width=2)  # Example parameters

if peak_indices is not None:
    print("Peak Indices:", peak_indices)
    print("Peak Properties:", peak_properties)

    import matplotlib.pyplot as plt
    plt.plot(df['Value'], label='Value')
    plt.plot(peak_indices, df['Value'][peak_indices], 'ro', label='Peaks')  # Mark peaks on the plot
    breakpoint()

#    # Annotate peaks with their properties (optional)
#    for i, peak_idx in enumerate(peak_indices):
#      plt.annotate(f"P:{peak_properties['prominences'][i]:.1f}, W:{peak_properties['widths'][i]:.1f}", 
#                   (peak_idx, df['Value'][peak_idx]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.legend()
    plt.show()
else:
  print("Invalid DataFrame or column name.")



data2 = {'Value': [10, 12, 15, 13, 16, 18, 17, 20, 19, 22, 20, 18, 15, 17, 21, 23, 22, 20, 18, 15],
        'OtherColumn': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
df2 = pd.DataFrame(data2)

peak_indices2, peak_properties2 = find_peaks_in_dataframe(df2, 'Value')  # Example parameters

if peak_indices2 is not None:
    print("Peak Indices:", peak_indices2)
    print("Peak Properties:", peak_properties2)

    import matplotlib.pyplot as plt
    plt.plot(df2['Value'], label='Value')
    plt.plot(peak_indices2, df2['Value'][peak_indices2], 'ro', label='Peaks')  # Mark peaks on the plot

    # Annotate peaks with their properties (optional)
    if 'prominences' in peak_properties2 and 'widths' in peak_properties2:
        for i, peak_idx in enumerate(peak_indices2):
          plt.annotate(f"P:{peak_properties2['prominences'][i]:.1f}, W:{peak_properties2['widths'][i]:.1f}", 
                       (peak_idx, df2['Value'][peak_idx]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.legend()
    plt.show()
else:
  print("Invalid DataFrame or column name.")

data3 = {'OtherColumn': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
df3 = pd.DataFrame(data3)

peak_indices3, peak_properties3 = find_peaks_in_dataframe(df3, 'Value')  # Example parameters

if peak_indices3 is not None:
    print("Peak Indices:", peak_indices3)
    print("Peak Properties:", peak_properties3)

    import matplotlib.pyplot as plt
    plt.plot(df3['Value'], label='Value')
    plt.plot(peak_indices3, df3['Value'][peak_indices3], 'ro', label='Peaks')  # Mark peaks on the plot

    # Annotate peaks with their properties (optional)
    if 'prominences' in peak_properties3 and 'widths' in peak_properties3:
        for i, peak_idx in enumerate(peak_indices3):
          plt.annotate(f"P:{peak_properties3['prominences'][i]:.1f}, W:{peak_properties3['widths'][i]:.1f}", 
                       (peak_idx, df3['Value'][peak_indices3]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.legend()
    plt.show()
else:
  print("Invalid DataFrame or column name.")
