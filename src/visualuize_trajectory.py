
import plotly.graph_objs as go
import pandas as pd
import pathlib


def read_data(path: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(path)

def visualize(data: pd.DataFrame) -> None:
    fig = go.Figure()
    print(data.columns)
    fig.add_trace(go.Scatter(x=data['px'], y=data['py'], mode="markers", text=data['# timestamp_in_seconds']))
    fig.add_trace(go.Scatter(x=data['qx'], y=data['qy'], mode="markers", text=data['qw']))
    fig.show()


if __name__ == '__main__':
    path = pathlib.Path("/home/amir/Desktop/pet_projects/internship/data/loza_visodom_0.csv")
    trajectory_data = read_data(path)
    visualize(trajectory_data)