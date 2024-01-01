import requests, json

BASE_URL = "https://hacker-news.firebaseio.com/v0"

def get_last_ten_stories() -> json:
    stories_ids: list[int] = requests.get(f"{BASE_URL}/topstories.json").json()[:10]
    stories: list = [get_story_by_id(id) for id in stories_ids]
    return stories


def get_story_by_id(id: str) -> json:
    return requests.get(f"{BASE_URL}/item/{id}.json").json()

def get_user_by_id(id: str) -> json:
    return requests.get(f"{BASE_URL}/user/{id}.json").json()

def parse_story(story: json) -> json:
    user = get_user_by_id(story['by'])
    return {
        'id': story['id'],
        'title': story['title'],
        'url': story['url'],
        'score': story['score'],
        'time': story['time'],
        'descendants': story['descendants'],
        'by': story['by'],
        'karma': user['karma']
    }


if __name__ == '__main__':
    stories = get_last_ten_stories()
    data: list = [parse_story(story) for story in stories]
    print(data)