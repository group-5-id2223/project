import feature_library as fl
import modal

LOCAL = False

if LOCAL == False:
    stub = modal.Stub('hackernews-score-pred')
    image = modal.Image.debian_slim().pip_install(['hopsworks'])

    @stub.function(image=image, 
                   schedule=modal.Period(days=1), 
                   secret=modal.Secret.from_name("hopsworks.ai"))
    def f():
        g()

def generate_live_data():
    stories = fl.get_top_ten_stories()
    data: list = [fl.process_story(story) for story in stories]
    return fl.convert_to_df(data)

def g():
    import hopsworks

    project = hopsworks.login(project="id2223_enric")
    fs = project.get_feature_store()

    hackernews_df = generate_live_data()

    hackernews_fg = fs.get_feature_group("hackernews_fg", 2)
    hackernews_fg.insert(hackernews_df, write_options={"wait_for_job": True})

if __name__ == '__main__':
    if LOCAL == True:
        g()
    else:
        modal.runner.deploy_stub(stub)