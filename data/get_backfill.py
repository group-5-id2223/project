import csv, json, concurrent.futures, argparse, random
from datetime import datetime
import feature_library as fl

FOLDER_PATH = '.'

def process_item(i: int) -> json:
    try:
        item: json = fl.get_item_by_id(i)
        if fl.is_story(item):
            return fl.process_story(item)
        return None
    except Exception as e:
        return None

def write_items_to_tsv(items: list[dict], file_name: str):
    with open(file_name, 'at') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(items[0].keys())
        for item in items:
            tsv_writer.writerow(item)

def format_number(number: int) -> str:
    if number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return str(number)

def generate_unique_random(min_val: int, max_val: int, existing_set: set) -> int:
    while True:
        rand_num = random.randint(min_val, max_val)
        if rand_num not in existing_set:
            return rand_num



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--rows', metavar='N', type=int, help='an integer for the number of rows',)
    parser.add_argument('--batch_size', metavar='N', type=int, help='an integer for the number of rows',
                        default=100)



    args = parser.parse_args()
    max_rows: int = fl.get_max_item()
    n_rows: int = args.rows if args.rows is not None else max_rows
    batch_size: int = args.batch_size

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M")
    base_file_name = f'{FOLDER_PATH}/data-{formatted_time}.tsv'
    max_item: int = fl.get_max_item()
    stories: list[dict] = []
    numbers: set[int] = set()
    written_items: int = 0
    
    while written_items < n_rows:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            while len(stories) < batch_size:
                if written_items >= n_rows:
                    break
                number = generate_unique_random(1, max_item, numbers)
                numbers.add(number)
                item = executor.submit(process_item, number)
                if item.result() is not None:
                    stories.append(item.result())
        written_items += batch_size
        write_items_to_tsv(stories, base_file_name)
        stories.clear()
        print(f"Processed {written_items} of {n_rows} - {datetime.now()}")
    print(f"Done. Time taken: {datetime.now() - now}")