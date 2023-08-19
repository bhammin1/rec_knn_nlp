import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import yaml
import os.path
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, ColumnsAutoSizeMode

import pandas as pd
import numpy as np
import news_rec as nr

# Help from https://github.com/mkhorasani/Streamlit-Authenticator?ref=blog.streamlit.io

history, cand, news = nr.all_preprocessing_web_app()
model_sts = nr.get_sts()


def display_candidates(user, history, candidate,news):
    '''
    Displays the news articles for a user to browse once logged in
    '''
    hist_news = history.loc[history["user_id"]== user]["news_id"].to_list()
    can_news = candidate.loc[candidate["user_id"]== user]["news_id"].to_list()
    exclude_news = hist_news + can_news

    news_to_display = news.loc[~news["news_id"].isin(exclude_news)]


    if news_to_display.shape[0] > 20:
        news_to_display = news_to_display[:20]

    return news_to_display

def add_new_user(new_user, can,copy_user="U9318"):
    '''
    Adds a new user to the databse
    '''
    global cand
    new_user_df = can.loc[can["user_id"]==copy_user].copy(deep=True)
    new_user_df["user_id"] = new_user
    can = pd.concat([can,new_user_df])
    cand = can.copy()

def update_history(user, hist, cand, new_reading):
    '''
    Adds records to a users reading history.
    '''
    global history
    fields = ["date", "news_id","category","sub_category","title"]
    
    for read in new_reading:
        # verify record is not already present
        if history.loc[(history["news_id"]== read) & (history["user_id"]== user)].shape[0] == 0:
            new_history = cand.loc[cand["news_id"]==read][fields].head(1)

            if new_history.shape[0] == 0:
                new_history = hist.loc[hist["news_id"]==read][fields].head(1)

            new_history["user_id"] = user
            hist = pd.concat([hist, new_history])
    history = hist.copy()

def recommend_sidebar(user, can, hist, news,sts):
    '''
    Produces recommendations on the side bar
    '''
    # filter to only the user
    user_can = can.loc[can["user_id"]==user]
    user_hist = hist.loc[hist["user_id"]==user]

    with st.sidebar:
        st.markdown('# Recommended Articles')
        if user_can.shape[0] == 0 or user_hist.shape[0] == 0:
            st.markdown("## No recommendations available yet")

        else:
            recs = nr.rec_any(user_can, user_hist , 3, "cosine","STS",sts, 1,True)

            if user in recs:
                recs_df = news.loc[news["news_id"].isin(recs[user])][["news_id","title","abstract"]]
                rec_titles = recs_df["title"].to_list()
                rec_abstract = recs_df["abstract"].to_list()
                
                for i in range(len(rec_titles)):
                    st.markdown("## Article "+ str(i+1))
                    st.markdown(rec_titles[i])
                    st.markdown("- "+ rec_abstract[i])

def recommender(name):
    '''
    Once loggin, the recommender system application launches
    This allows for recommendations to be created while the user
    browses the news stand
    '''
    # add user if no history exists
    if cand.loc[cand["user_id"] == name].shape[0] == 0:

        add_new_user(name, cand)

        

    display_list = display_candidates(name, history, cand,news)

    st.write(f'Welcome', name)
    st.title('Browse the News Stand')
    st.write('Please select the articles that you would like to read')


    # select the columns you want the users to see
    gb = GridOptionsBuilder.from_dataframe(display_list[["news_id", "title"]])
    # configure selection
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    gridOptions = gb.build()

    data = AgGrid(display_list[["news_id","title"]],
              gridOptions=gridOptions,
              enable_enterprise_modules=True,
              allow_unsafe_jscode=True,
              update_mode=GridUpdateMode.SELECTION_CHANGED,
              columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
    selected_rows = data["selected_rows"]

    if len(selected_rows) > 0:
        # add record to your history
        new_history = [article["news_id"] for article in selected_rows]
        update_history(name, history, cand, new_history)

    # get recommendations
    recommend_sidebar(name, cand, history, news, model_sts)



def save_user(all_creds):
    '''
    Save the new user along with all other users
    Parameter:
    all_creds: dict of all users registered in the system including the new addition
    '''

    with open('./configure_new.yaml', "w") as outfile:
        yaml.dump(all_creds, outfile)





def register_new_user(authenticator):
    '''
    Creates a register user session
    '''
    try:
        if authenticator.register_user('Register user', preauthorization=False):
            save_user(authenticator.credentials)
            st.success('User registered successfully')
            st.write("Please return to Login Menu")

            
    except Exception as e:
        st.error(e)

def login(authenticator):
    '''
    Logins a user to the system
    '''

    #check if a new user was created
    if os.path.isfile('./configure_new.yaml'):
        with open('./configure_new.yaml') as file:
            new_cred = yaml.load(file, Loader=SafeLoader)
        authenticator.credentials = new_cred

    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        authenticator.logout('Logout', 'main')
        recommender(name)
    elif authentication_status == False:
        st.error('Username/password is incorrect')

    elif authentication_status == None:
        st.warning('Please enter your username and password')



def start_app():
    '''
    Starting up the application
    '''

    st.title("Cyber News Cafe")

    menu = ["About","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    with open('./configure.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )


    if choice == "About":
        st.subheader("About")
        st.write(f'Welcome To the Cyber News Cafe')
        st.write(f'Please log in or create an account to access the news stand')
        st.write(f'Once logged in, you will receive personal recommendations after viewing one article')
        
    elif choice == "Login":
        login(authenticator)
    elif choice == "SignUp":
        register_new_user(authenticator)



if __name__ == "__main__":
    start_app()
