import requests, json
import pandas as pd
import numpy as np

BASE_URL = "https://hacker-news.firebaseio.com/v0"

def get_last_ten_stories() -> json:
    stories: list = []
    stories_ids: list[int] = requests.get(f"{BASE_URL}/newstories.json").json()
    i = 0
    for story_id in stories_ids:
        if is_story(get_item_by_id(story_id)):
            stories.append(get_item_by_id(story_id))
        if len(stories) >= 10:
            break
    return stories

def get_max_item() -> int:
    return requests.get(f"{BASE_URL}/maxitem.json").json()

def get_item_by_id(id: str) -> json:
    return requests.get(f"{BASE_URL}/item/{id}.json").json()

def is_story(item: json) -> bool:
    return item['type'] == 'story'

def get_user_by_id(id: str) -> json:
    return requests.get(f"{BASE_URL}/user/{id}.json").json()

def process_story(story: json) -> json:
    user = get_user_by_id(story['by'])

    return {
        'id': story['id'],
        'title': story['title'],
        'url': story.get('url', ''),
        'score': story['score'],
        'time': story['time'],
        'descendants': story['descendants'],
        'by': story['by'],
        'karma': user['karma']
    }

def convert_to_df(stories: list) -> pd.DataFrame:
    data = pd.DataFrame(stories)
    data['id'] = data['id'].astype(np.int8)
    data['score'] = data['score'].astype(float)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    return pd.DataFrame(data)