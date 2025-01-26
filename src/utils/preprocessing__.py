import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch 


def preprocess(df, lag_hours=3, test_size=0.2):
    """
    Preprocess the bike-sharing data for time series prediction.
    
    This function prepares the dataset by adding lag features and creating the target variable.
    It then splits the data into training and testing sets based on the specified test size.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing bike-sharing data. It should include columns:
        - 'start_station_id': Station identifier.
        - 'started_at_hour': Hour at which the bike ride started.
        - 'rides_': Number of rides (demand).
    
    lag_hours : int, optional (default=3)
        The number of lag hours to consider for creating lag features. 
        A lag hour represents the number of past hours' demand to predict future demand.
    
    test_size : float, optional (default=0.2)
        Proportion of the data to be used as the test set. The remaining data is used for training.
    
    Returns:
    --------
    X_train : np.ndarray
        The training set features (lagged demand values).
    
    y_train : np.ndarray
        The training set target values (next hour's demand).
    
    X_test : np.ndarray
        The testing set features (lagged demand values).
    
    y_test : np.ndarray
        The testing set target values (next hour's demand).
    
    Notes:
    ------
    - The function assumes the input dataframe is sorted by 'start_station_id' and 'started_at_hour'.
    - Lag features are created using the past 'lag_hours' number of hours' demand to predict future demand.
    - The target variable is the demand for the next hour (shifted by one time step).
    - Missing values are dropped for the lag features and target variable.
    """
    

    df = df.sort_values(by=['start_station_id', 'started_at_hour'])
    #add lags
    for i in range(1, lag_hours + 1):
        df[f'lag_{i}'] = df.groupby('start_station_id')['rides_'].shift(i)


    df.dropna(subset=[f'lag_{i}' for i in range(1, lag_hours + 1)], inplace=True)


    df['target'] = df.groupby('start_station_id')['rides_'].shift(-1)
    df.dropna(subset=['target'], inplace=True)


    X = df[[f'lag_{i}' for i in range(1, lag_hours + 1)]].values
    y = df['target'].values


    train_size = int(len(X) * (1 - test_size))

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test




def preprocess_graph_data(df):
    """
    Preprocesses the graph data from a DataFrame and prepares the node features 
    and adjacency matrix for a graph neural network (GCN) model.
    
    This function extracts relevant features, normalizes them, creates a mapping 
    for station IDs to integer indices, and constructs the adjacency matrix 
    representing the connections (edges) between stations.
    
    Args:
        df (DataFrame): A pandas DataFrame containing ride data with columns 
                        such as 'start_lat', 'start_lng', 'end_lat', 'end_lng', 
                        'rides_', 'start_station_id', 'end_station_id'.
    
    Returns:
        torch.Tensor: The normalized node features (x), of shape (num_nodes, num_features).
        torch.Tensor: The edge index for the graph (edge_index), of shape (2, num_edges).
    
    Notes:
        - The adjacency matrix is constructed based on the connection between stations 
          (if a ride starts and ends at two different stations, an edge is created).
        - The adjacency matrix is assumed to represent an undirected graph.
        - The feature normalization is done using `StandardScaler`.
    """
    

    features = df[['start_lat', 'start_lng', 'end_lat', 'end_lng', 'rides_']].values
    features = StandardScaler().fit_transform(features)  


    station_ids = pd.concat([df['start_station_id'], df['end_station_id']]).unique()
    station_map = {station_id: idx for idx, station_id in enumerate(station_ids)}
    
 
    num_stations = len(station_ids)  
    adj_matrix = np.zeros((num_stations, num_stations), dtype=int)
    

    for _, row in df.iterrows():
        start_station = station_map[row['start_station_id']]
        end_station = station_map[row['end_station_id']]
        

        adj_matrix[start_station, end_station] = 1
        adj_matrix[end_station, start_station] = 1
    

    edge_index = torch.tensor(np.nonzero(adj_matrix), dtype=torch.long)


    x = torch.tensor(features, dtype=torch.float32)
    
    return x, edge_index
