import feature_library as fl

if __name__ == '__main__':
    stories = fl.get_last_ten_stories()
    data: list = [fl.process_story(story) for story in stories]
    print(data)