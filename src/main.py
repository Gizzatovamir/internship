import pathlib
import sqlite3
from src.read_log import read_one_msg

if __name__ == '__main__':
    path: pathlib.Path = pathlib.Path("/home/amir/Desktop/pet_projects/internship/data/23_07_08_visual_odometry_0_0.db3")
    db = sqlite3.connect(path.as_posix())
    read_one_msg(db.cursor())