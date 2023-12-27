import csv, requests, json, concurrent.futures, argparse
from datetime import datetime

FOLDER_PATH = '.'
BASE_URL = "https://hacker-news.firebaseio.com/v0"

def get_max_item() -> int:
    return requests.get(f"{BASE_URL}/maxitem.json").json()

def get_post_by_id(id: str) -> json:
    return requests.get(f"{BASE_URL}/item/{id}.json").json()

def is_story(item: json) -> bool:
    return item['type'] == 'story'

def get_user_by_id(id: str) -> json:
    return requests.get(f"{BASE_URL}/user/{id}.json").json()


def process_item(i: int) -> json:
    item: json = get_post_by_id(i)
    try:
        if is_story(item):
            user = get_user_by_id(item['by'])
            return [item['id'], item['title'], item['url'], item['score'], item['time'], item['descendants'], item['by'], user['karma']]
    except:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('rows', metavar='N', type=int, help='an integer for the number of rows')

    args = parser.parse_args()
    n_rows: int = args.rows

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M")
    file_name = f'{FOLDER_PATH}/data-{formatted_time}.tsv'
    max_item: int = get_max_item()
    items: list = []
    written_items: int = 0
    
    while written_items < n_rows:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for i in range(written_items, min(written_items + 100, n_rows)):
                item = executor.submit(process_item, i)
                if item.result() is not None:
                    items.append(item.result())
        for item in items:
            with open(file_name, 'at') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(item)
        written_items += len(items)
        items = []
        print(f"Processed {written_items} of {n_rows} - {datetime.now()}")
    print(f"Done. Time taken: {datetime.now() - now}")