from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 64
    train_batch_size = 1
    eval_batch_size = 1
    num_epochs = 3
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = num_epochs / 10 #500
    mixed_precision = "no"#"fp16"
    output_dir = "logs"
    save_image_epochs = num_epochs # Only save on the last epoch for now

    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0

@dataclass
class FotmobScraperConfig:
    logging = False
    time_out = 10 # seconds to wait for element to load before timing out
    fotmob_url = 'https://www.fotmob.com/'
    missing_crest_redirect = "/_next/static/media/team_fallback.3ae01170.png" # if no crest data then you'll get redirected here
    driver_path = './chromedriver' # TODO Fix path

    # Main page CSS elements
    league_wrapper = 'MuiCollapse-wrapperInner MuiCollapse-vertical css-8atqhb'
    league_elements =  'css-15d5019-LeagueListHeaderButton edx0mqc2'

    # League Overiew CSS elements
    header_elements = "css-1jc3wnl-NavContainerCSS e17ala8t1"
    header_elemnts_CSS = ".css-1jc3wnl-NavContainerCSS.e17ala8t1" # TODO just derive this from the above

    # League Table CSS elements
    table_element = "css-53qoct-TableWrapper ecspc020"
    table_element_CSS = ".css-53qoct-TableWrapper.ecspc020" # TODO just derive this from the above
    team_cell_element = "TeamName css-mxvysr-TeamName-teamNameStyle e13kov182"













