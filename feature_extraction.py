from bs4 import BeautifulSoup
import os
import features as fe
# import pandas as pd

# 1 DEFINE A FUNCTION THAT OPENS A HTML FILE AND RETURNS THE CONTENT
file_name = "mini_dataset/9.html"


def open_file(f_name):
    with open(f_name, "r") as f:
        return f.read()


# 2 DEFINE A FUNCTION THAT CREATES A BEATIFULSOUP OBJECT
def create_soup(text):
    return BeautifulSoup(text, "html.parser")


# 3 DEFINE A FUNCTION THAT CREATES A VECTOR BY RUNNING ALL FEATURE FUNCTIONS FOR THE SOUP OBJECT
def create_vector(soup):
    return [
        fe.has_title(soup),
        fe.has_input(soup),
        fe.has_button(soup),
        fe.has_image(soup),
        fe.has_submit(soup),
        fe.has_link(soup),
        fe.has_password(soup),
        fe.has_email_input(soup),
        fe.has_hidden_element(soup),
        fe.has_audio(soup),
        fe.has_video(soup),
        fe.number_of_inputs(soup),
        fe.number_of_buttons(soup),
        fe.number_of_images(soup),
        fe.number_of_option(soup),
        fe.number_of_list(soup),
        fe.number_of_TH(soup),
        fe.number_of_TR(soup),
        fe.number_of_href(soup),
        fe.number_of_paragraph(soup),
        fe.number_of_script(soup),
        fe.length_of_title(soup),
        fe.has_h1(soup),
        fe.has_h2(soup),
        fe.has_h3(soup),
        fe.length_of_text(soup),
        fe.number_of_clickable_button(soup),
        fe.number_of_a(soup),
        fe.number_of_img(soup),
        fe.number_of_div(soup),
        fe.number_of_figure(soup),
        fe.has_footer(soup),
        fe.has_form(soup),
        fe.has_text_area(soup),
        fe.has_iframe(soup),
        fe.has_text_input(soup),
        fe.number_of_meta(soup),
        fe.has_nav(soup),
        fe.has_object(soup),
        fe.has_picture(soup),
        fe.number_of_sources(soup),
        fe.number_of_span(soup),
        fe.number_of_table(soup)
    ]


# 4 RUN STEP 1,2,3 FOR ALL HTML FILES AND CREATE A 2-D ARRAY
folder = "mini_dataset"


def create_2d_list(folder_name):
    directory = os.path.join(os.getcwd(), folder_name)
    data = []
    for file in sorted(os.listdir(directory)):
        soup = create_soup(open_file(directory + "/" + file))
        data.append(create_vector(soup))
    return data

"""
# 5 CREATE A DATAFRAME BY USING 2-D ARRAY
data = create_2d_list(folder)

columns = [
    'has_title',
    'has_input',
    'has_button',
    'has_image',
    'has_submit',
    'has_link',
    'has_password',
    'has_email_input',
    'has_hidden_element',
    'has_audio',
    'has_video',
    'number_of_inputs',
    'number_of_buttons',
    'number_of_images',
    'number_of_option',
    'number_of_list',
    'number_of_th',
    'number_of_tr',
    'number_of_href',
    'number_of_paragraph',
    'number_of_script',
    'length_of_title',
    'has_h1',
    'has_h2',
    'has_h3',
    'length_of_text',
    'number_of_clickable_button',
    'number_of_a',
    'number_of_img',
    'number_of_div',
    'number_of_figure',
    'has_footer',
    'has_form',
    'has_text_area',
    'has_iframe',
    'has_text_input',
    'number_of_meta',
    'has_nav',
    'has_object',
    'has_picture',
    'number_of_sources',
    'number_of_span',
    'number_of_table'
]

df = pd.DataFrame(data=data, columns=columns)

print(df.head(5))
"""






















