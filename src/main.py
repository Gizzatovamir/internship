import pathlib
import sqlite3
from src.read_log import read_one_msg
from src.gen_depth_data import gen_depth_data
from src.query import query_get_one_msg
from rosbags.serde import deserialize_cdr
import utils.constants as constants


# utf8 codec error
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 133: invalid continuation byte


def get_utm_pos(_db: sqlite3.Cursor, destination_path: pathlib.Path) -> None:
    timestamp, utm_ros_msg = _db.execute(query_get_one_msg(constants.UTM_TOPIC_ID)).fetchone()
    utm = deserialize_cdr(utm_ros_msg, "nav_msgs/msg/Odometry")
    print(timestamp, utm)


if __name__ == "__main__":
    # path: pathlib.Path = pathlib.Path("/home/amir/Desktop/pet_projects/internship/data/23_07_08_visual_odometry_0_0.db3")
    path: pathlib.Path = pathlib.Path(
        "/home/amir/Desktop/pet_projects/internship/data/velodyne-laba-circle/2023_12_18_18_15_33_0.db3"
    )
    db = sqlite3.connect(path.as_posix())
    cursor = db.cursor()
    get_utm_pos(cursor, pathlib.Path("utm_data"))
    # gen_depth_data(
    #     db.cursor(),
    #     pathlib.Path("./eval_depth/"),
    #     query_get_one_msg(LIDAR_POINTS_TOPIC_ID),
    # )
