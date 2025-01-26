import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')


def handle_categorical_data(df, col, n=0, hue=None):

    """ 
    This function plot freq of each type of categorical feature.

    -------
    Params:
    -------
    df: Dataframe
    col: str (categorical col name)


    ------
    return:
    ------
       sns barplot object
    
    """
    if n > 0:
        x = df[col].value_counts().head(n)
    elif n < 0: 
        x = df[col].value_counts().tail(-n)
    else:
        x = df[col].value_counts()
    x_vcount = pd.DataFrame({col: x.index, 'Count': x.values})
    plt = sns.barplot(x=x_vcount[col] , y=x_vcount['Count'] ,palette='viridis',hue=hue)


    return plt 

def grid_plot(df, plt_type, target, feature):
    """
    This function plot relatioship between categorical feature and other one (bivariate).

    -------
    @Params:
    -------
    df: DataFrame
    plt_type: plot function (sns or ...)
    target: str col name 
    feature: str col name 


    -------
    return:
    -------
    bivariate plot based on plot type  (grid)
    
    """
    fig = sns.FacetGrid(df, col=target)
    fig.map(plt_type,feature)
    
    return fig 



def categorize_time_of_day(dt):
    """
    Categorizes the time of day based on the given datetime object.
    """
    hour = dt.hour
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 20:
        return 'Evening'
    else:
        return 'Night'
    

def plot_station_trends(data, n=5, top=True,markers='o', station_col='start_station_id', time_col='started_at_hour', rides_col='rides_'):
    """
    Plots the ride trends for the top or lowest N stations based on total rides.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame containing station data.
        n (int): The number of stations to plot (default: 5).
        top (bool): Whether to plot the top N (True) or lowest N (False) stations (default: True).
        station_col (str): The column representing station IDs.
        time_col (str): The column representing the time (hourly).
        rides_col (str): The column representing the number of rides.
    """
    # Convert time column to datetime if not already
    data[time_col] = pd.to_datetime(data[time_col])

    # Aggregate total rides per station
    station_totals = data.groupby(station_col)[rides_col].sum().reset_index()
    
    # Select top or lowest N stations
    if top:
        selected_stations = station_totals.nlargest(n, rides_col)[station_col]
    else:
        selected_stations = station_totals.nsmallest(n, rides_col)[station_col]

    # Filter the data for the selected stations
    filtered_data = data[data[station_col].isin(selected_stations)]
    
    # Plotting
    plt.figure(figsize=(14, 8))
    sns.lineplot(
        data=filtered_data, 
        x=time_col, 
        y=rides_col, 
        hue=station_col, 
        marker=markers, 
        palette='tab10'
    )
    
    # Add titles and labels
    direction = 'Top' if top else 'Lowest'
    plt.title(f'{direction} {n} Stations: Number of Rides Over Time', fontsize=16)
    plt.xlabel('Time (Hourly)', fontsize=14)
    plt.ylabel('Number of Rides', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.legend(title='Station ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
