def query_get_one_msg(topic_id: int) -> str:
    return f"""SELECT timestamp, data from messages WHERE {topic_id} == topic_id"""