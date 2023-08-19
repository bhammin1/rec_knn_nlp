import pandas as pd
import numpy as np
from scipy import spatial
from scipy import stats
import os
import os.path
import csv
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

def load_tsv(file, cols):
    '''
    Given a path and list of columns,
    loads a tsv file into a pandas data frame
    '''
    df = pd.read_table(file,sep="\t", header=None, names=cols)

    return df

def one_hot(df, col,vals):
    '''
    Performs one hot-encoding on a single column

    Parameters:
    df: pandas dataframe
    col: string for column name to convert encoding
    vals: list of values for a column
    '''
    for i, rw in df.iterrows():
        encoding = []
        current_val = rw[col]
        # create one-hot encoding per row
        for v in vals:
            if v == current_val:
                encoding.append(1)
            else:
                encoding.append(0)

        # update the df
        df.at[i, col] = encoding

def create_rows(df, col):
    '''
    Converts a list field into multiple rows
    Each row contains one value of the list

    Parameters:
    df: pandas df
    col: string representing the string name
    '''
    # first convert string into a list/array
    df[col] = df[col].str.split()

    # explored into multiple rows
    new_df = df.explode(col)
    return new_df

def get_similarity(candidate, history,cols, distance):
    ''' 
    Takes a single record for candidate and history
    List of cols to take distance on
    Performs distance calc on each feature
    returns the sum of the distance

    Parameters:
    candidate: pd df for candidate articles (should be contain one row)
    history: pd df for history articles (should be contain one row)
    cols: list of cols to measure distance between
    distance: type of distance measure cosine or pearson
    '''
    sim_score = 0
    for c in cols:
        h_col = history[c].iloc[0]
        c_col = candidate[c].iloc[0]
        
        if distance == "cosine":
            # negate because smaller number is better. 
            # But we will take the max in next step
            sim_score -= spatial.distance.cosine(h_col, c_col)
        else: # pearson
            sim_score += stats.pearsonr(h_col, c_col).statistic
    return sim_score

def get_similarity_semantic(candidate, history,cols, distance, model):
    ''' 
    Takes a single record for candidate and history
    List of cols to take distance on
    Performs distance calc on each feature
    returns the sum of the distance
    Then adds sts score to the distance

     Parameters:
    candidate: pd df for candidate articles (should be contain one row)
    history: pd df for history articles (should be contain one row)
    cols: list of cols to measure distance between
    distance: type of distance measure cosine or pearson
    model: STS model 
    '''
    sim_score = get_similarity(candidate, history,cols, distance)

    # get titles
    title1 = history["title"].iloc[0]
    title2 = candidate["title"].iloc[0]
    
    # perform semantic text similarity
     #Compute embedding for both lists
    embedding_1= model.encode(title1, convert_to_tensor=True)
    embedding_2 = model.encode(title2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2) # tensor returned

    sim_score += score.numpy()[0][0]

    return sim_score

def get_similarity_entailement(candidate, history,cols, distance, model,direction):
    ''' 
    Takes a single record for candidate and history
    List of cols to take distance on
    Performs distance calc on each feature
    returns the sum of the distance
    Then adds the entailement score to the distance

    Parameters:
    candidate: pd df for candidate articles (should be contain one row)
    history: pd df for history articles (should be contain one row)
    cols: list of cols to measure distance between
    distance: type of distance measure cosine or pearson
    model: NLI model
    direction: number of times to generate entailement score
    '''
    sim_score = get_similarity(candidate, history,cols, distance)

    # get titles
    title1 = history["title"].iloc[0]
    title2 = candidate["title"].iloc[0]
    
    # perform nli
    if direction == 1:
        nli_scores2 = model.predict((title2, title1))
        sim_score +=  nli_scores2[1] 

    else:
        nli_scores1 = model.predict((title1, title2))
        nli_scores2 = model.predict((title2, title1))

        sim_score += nli_scores1[1] + nli_scores2[1]   # index 1 is entailement score

    return sim_score

def knn_per_history(candidates, scores, k):
    '''
    Given a candidate df with scores for each record
    Return the top k scored records
    '''
    candidates["scores"] = scores # append scores
    candidates = candidates.sort_values("scores", ascending = False) # top  scores on top
    topK = candidates.head(k)
    return topK

def recommend_one(topKs):
    ''''
    Recommends the top k (only highest)
    '''
    recs = pd.concat(topKs)

    # Group by count
    recs_gr = recs.groupby(["news_id"])["news_id"].size().reset_index(name='counts') 

    # find max count, store those value
    recs_max = recs_gr[recs_gr['counts']==recs_gr['counts'].max()]
    recs_max_list = recs_max["news_id"].to_list()

    # if tie, go by score
    if len(recs_max_list) > 1:
        recs_tie = recs[recs["news_id"].isin(recs_max_list)]
        recs_tie = recs_tie.sort_values(by=["scores", "title"], ascending=False)
        recs_max_list = recs_tie["news_id"].to_list()

    return recs_max_list[0]

def recommend_k(topKs,k=5):
    '''
    Recommends K articles
    '''
    final_recs = []
    # combine list of dfs with historical top k
    recs = pd.concat(topKs)

    # Group by count
    recs_gr = recs.groupby(["news_id"])["news_id"].size().reset_index(name='counts') 

    # order the counts
    top_counts = recs_gr["counts"].unique()
    top_counts = -np.sort(-top_counts) # make in descending order
    
    # populate final recs with tie breaker logic
    while len(final_recs) < k:
        if top_counts.size == 0: # error handling
            break
        cnt = top_counts[0]
        recs_max = recs_gr[recs_gr['counts']==cnt]
        recs_max_list = recs_max["news_id"].to_list()

        if len(recs_max_list) > 1:
            recs_tie = recs[recs["news_id"].isin(recs_max_list)]
            recs_tie = recs_tie.sort_values(by=["scores", "title"], ascending=False)
            recs_max_list = recs_tie["news_id"].to_list()
        
        for r in recs_max_list:
            if r not in final_recs and len(final_recs) < k:
                final_recs.append(r)

        # move up to next count
        top_counts = np.delete(top_counts,0)

    return final_recs

def rec_any(candidate, history, k, distance,exp,model, directions=1,debug=False):
    '''
    Provides recommendations using NN only approach

    Parameters:
    candidate: candidate pd DF
    history: history panda DF
    k: number of articles to recommend per user
    distance: distance measure to use: cosine or pearson
    exp: experiment name
    model: NLP model
    direction: NLI experiment number of entailement score
    debug: bool indicator to print status
    '''
    users = history["user_id"].unique()
    user_pred = {}
    for user in users:
        user_can =  candidate.loc[candidate["user_id"] == user]
        user_hist =  history.loc[history["user_id"] == user]

        # get list of historical articles
        hist_recs = user_hist["news_id"].unique()
        candidate_recs = user_can["news_id"].unique()
        user_topK = []

        #score each historical article to candidate article
        for hr in hist_recs:
            hr_row = user_hist.loc[user_hist["news_id"] == hr]
            candidate_score = []
            for cr in candidate_recs:
                cr_row = user_can.loc[user_can["news_id"]== cr]
                if(cr_row.shape[0] != 1):
                    print("Wrong assumption")
                    print(cr_row.shape)
                    print(cr_row)
                
                if exp == 'NN':
                    score = get_similarity(cr_row, hr_row, ["category","sub_category"], distance)
                elif exp == 'STS':
                    score = get_similarity_semantic(cr_row, hr_row, ["category","sub_category"], distance, model)
                else: # entailement
                    score = get_similarity_entailement(cr_row, hr_row, ["category","sub_category"], distance, model,directions)
                candidate_score.append(score)
            # find top K candidate per historical article
            user_can_c = user_can.copy() # to avoid a warning
            history_topK = knn_per_history(user_can_c, candidate_score, k)
            user_topK.append(history_topK)
        # recommend 5 articles based on users top K
        preds = recommend_k(user_topK)
        if len(preds) == 5: # dont include bad records
            user_pred[user] = preds

        if debug:
            print("User Status",np.where(users == user), "/", users.shape)
    return user_pred
'''
def rec(candidate, history, k, distance):
    users = history["user_id"].unique()
    user_pred = {}
    for user in users:
        user_can =  candidate.loc[candidate["user_id"] == user]
        user_hist =  history.loc[history["user_id"] == user]

        # get list of historical articles
        hist_recs = user_hist["news_id"].unique()
        candidate_recs = user_can["news_id"].unique()
        user_topK = []

        #score each historical article to candidate article
        for hr in hist_recs:
            hr_row = user_hist.loc[user_hist["news_id"] == hr]
            candidate_score = []
            for cr in candidate_recs:
                cr_row = user_can.loc[user_can["news_id"]== cr]
                if(cr_row.shape[0] != 1):
                    print("Wrong assumption")
                    print(cr_row.shape)
                    print(cr_row)
                score = get_similarity(cr_row, hr_row, ["category","sub_category"], distance)
                candidate_score.append(score)
            # find top K candidate per historical article
            user_can_c = user_can.copy() # to avoid a warning
            history_topK = knn_per_history(user_can_c, candidate_score, k)
            user_topK.append(history_topK)
        # recommend one article based on users top K
        user_pred[user] = recommend_one(user_topK)
    return user_pred


def rec_sts(candidate, history, k, distance,model_sts, debug=False):
    users = history["user_id"].unique()
    user_pred = {}
    for user in users:
        user_can =  candidate.loc[candidate["user_id"] == user]
        user_hist =  history.loc[history["user_id"] == user]
        print("User Can shape",user_can.shape)
        print("User His shape",user_hist.shape)

        # get list of historical articles
        hist_recs = user_hist["news_id"].unique()
        candidate_recs = user_can["news_id"].unique()
        user_topK = []

        print("Art Can shape",candidate_recs.shape)
        print("art His shape",hist_recs.shape)

        #score each historical article to candidate article
        for hr in hist_recs:
            hr_row = user_hist.loc[user_hist["news_id"] == hr]
            candidate_score = []
            for cr in candidate_recs:
                cr_row = user_can.loc[user_can["news_id"]== cr]
                if(cr_row.shape[0] != 1):
                    print("Wrong assumption")
                    print(cr_row.shape)
                    print(cr_row)
                score = get_similarity_semantic(cr_row, hr_row, ["category","sub_category"], distance, model_sts)
                candidate_score.append(score)
            # find top K candidate per historical article
            user_can_c = user_can.copy() # to avoid a warning
            history_topK = knn_per_history(user_can_c, candidate_score, k)
            user_topK.append(history_topK)
        # recommend one article based on users top K
        user_pred[user] = recommend_one(user_topK)

        if debug:
            print("User Status",np.where(users == user), "/", users.shape)
    return user_pred

def rec_entailement(candidate, history, k, distance,model_nli,debug=False, directions=1):
    users = history["user_id"].unique()
    user_pred = {}
    for user in users:
        user_can =  candidate.loc[candidate["user_id"] == user]
        user_hist =  history.loc[history["user_id"] == user]

        # get list of historical articles
        hist_recs = user_hist["news_id"].unique()
        candidate_recs = user_can["news_id"].unique()
        user_topK = []

        #score each historical article to candidate article
        for hr in hist_recs:
            hr_row = user_hist.loc[user_hist["news_id"] == hr]
            candidate_score = []
            for cr in candidate_recs:
                cr_row = user_can.loc[user_can["news_id"]== cr]
                if(cr_row.shape[0] != 1):
                    print("Wrong assumption")
                    print(cr_row.shape)
                    print(cr_row)
                score = get_similarity_entailement(cr_row, hr_row, ["category","sub_category"], distance, model_nli,directions)
                candidate_score.append(score)
            # find top K candidate per historical article
            user_can_c = user_can.copy() # to avoid a warning
            history_topK = knn_per_history(user_can_c, candidate_score, k)
            user_topK.append(history_topK)
        # recommend one article based on users top K
        user_pred[user] = recommend_one(user_topK)

        if debug:
            print("User Status",np.where(users == user), "/", users.shape)
    return 
'''
def get_metrics(candidate, preds:dict):
    '''
    Returns the accuracy of the model
    candidate: pd dataframe with label of candidate articles
    preds: dictionary of predicted user recommendations
    '''
    preds_series = pd.Series(preds, name="prediction")
    pred_df = preds_series.to_frame()
    pred_df["user_id"] = pred_df.index

    results_df = candidate.merge(pred_df, left_on =["user_id","news_id"], right_on =["user_id","prediction"] )

    correct_preds = results_df.loc[results_df["label"]=='1'].shape[0]
    accuracy = correct_preds/results_df.shape[0]

    return accuracy

def dict_to_csv(result, file):
    '''
    Saves the prediction output to a csv
    '''
    df_result = pd.DataFrame(data=result.items(), columns=["user_id","preds"])
    df_result[['pred_1','pred_2','pred_3','pred_4','pred_5']] = pd.DataFrame(df_result["preds"].tolist(), index= df_result.index)
    
    #result_series = pd.Series(result, name="prediction")
    #result_series.to_csv(file, index_label ="user_id")
    df_result.to_csv(file, index=False)

def result_dict(file):
    '''
    Reads the result csv and returns a dictionary
    '''
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]:rows[1] for rows in reader}
    mydict.pop('user_id', None)

    return mydict

def all_preprocessing():
    '''
    Preforms all pre-processing on the MIND data
    '''
    # load data
    behav_cols = ["impression_id", "user_id","time","history","impressions"]
    news_cols = ["news_id","category","sub_category","title","abstract","url","title_entities","abstract_entitites"]
    b_df = load_tsv("./data/behaviors.tsv", behav_cols)
    news_df = load_tsv("./data/news.tsv", news_cols)

    b_df["date"] = pd.to_datetime(b_df["time"]).dt.date

    # reduce date to only the last date
    b_df = b_df.loc[b_df["date"].astype(str) == "2019-11-14"]

    # one hot encoding
    # update category and subcategory
    news_cats = news_df.groupby(["category"])["category"].count()
    cats = list(news_cats.index)

    subcats = news_df.groupby(["sub_category"])["sub_category"].count()
    subcats = list(subcats.index)

    one_hot(news_df, "category",cats)
    one_hot(news_df, "sub_category",subcats)

    # filter out multiple impressions per day
    b_df['RN'] = b_df.sort_values(['time'], ascending=[False]).groupby(['user_id',"date"]).cumcount() + 1
    #b_df.loc[b_df["user_id"] == 'U79549'].sort_values(['time']) # check work
    b_df= b_df.loc[b_df["RN"] == 1] 


    # make history and cand data
    # History Will be the users historical articles combined with news relating info
    history_df_cols = [ "user_id","date","history"]
    candidate_df_cols = [ "user_id","date","impressions"]

    history_df = b_df[history_df_cols].copy(deep=True)
    candidate_df = b_df[candidate_df_cols].copy(deep=True)

    # split data into rows
    candidate_df = create_rows(candidate_df, "impressions")
    history_df = create_rows(history_df, "history")
    history_df = history_df.rename(columns={"history":"news_id"}) # rename history column for joining to news df

    # MAKE LABEL 
    candidate_df[["news_id","label"]] =candidate_df["impressions"].str.split(pat="-", expand=True)

    # join news data
    candidate_df = pd.merge(candidate_df,news_df, on ="news_id").drop(columns=
                                                                    ["impressions","abstract","url","title_entities",	"abstract_entitites"])
    history_df= pd.merge(history_df,news_df, on ="news_id").drop(columns=
                                                                    ["abstract","url","title_entities",	"abstract_entitites"])
    
    test_can = candidate_df.loc[candidate_df["date"].astype(str) == "2019-11-14"]
    test_history = history_df.loc[history_df["date"].astype(str) == "2019-11-14"]

    #remove too many users

    can_cnt = test_can.groupby(["user_id"]).size().reset_index(name='counts')
    hist_cnt = test_history.groupby(["user_id"]).size().reset_index(name='counts')
    few_cans = can_cnt.loc[(can_cnt["counts"]< 11) & (can_cnt["counts"]> 5)]
    few_hist = hist_cnt.loc[hist_cnt["counts"]< 11]
    common_users = set(few_cans["user_id"].to_list()).intersection(set(few_hist["user_id"].to_list()))
    common_users = list(common_users)

    #hist_sample = test_history.loc[test_history["user_id"].isin(common_users[:1000])] # replaced users
    #can_sample = test_can.loc[test_can["user_id"].isin(common_users[:1000])] # replaced users
    print(test_history.shape, test_can.shape)

    return test_history, test_can


def all_preprocessing_final():
    '''
    Preforms all pre-processing on the MIND data
    This is for the final experiments only
    Different than above function because this processing doesnt filter out by date

    '''
    # load data
    behav_cols = ["impression_id", "user_id","time","history","impressions"]
    news_cols = ["news_id","category","sub_category","title","abstract","url","title_entities","abstract_entitites"]
    b_df = load_tsv("./data/behaviors.tsv", behav_cols)
    news_df = load_tsv("./data/news.tsv", news_cols)

    b_df["date"] = pd.to_datetime(b_df["time"]).dt.date

    # reduce date to only the last date
    #b_df = b_df.loc[b_df["date"].astype(str) == "2019-11-14"]

    news_df["category_str"] = news_df["category"]
    news_df["sub_category_str"] = news_df["sub_category"]

    # one hot encoding
    # update category and subcategory
    news_cats = news_df.groupby(["category"])["category"].count()
    cats = list(news_cats.index)

    subcats = news_df.groupby(["sub_category"])["sub_category"].count()
    subcats = list(subcats.index)

    one_hot(news_df, "category",cats)
    one_hot(news_df, "sub_category",subcats)


    # filter out multiple impressions per day
    b_df['RN'] = b_df.sort_values(['time'], ascending=[False]).groupby(['user_id',"date"]).cumcount() + 1
    #b_df.loc[b_df["user_id"] == 'U79549'].sort_values(['time']) # check work
    b_df= b_df.loc[b_df["RN"] == 1] 


    # make history and cand data
    # History Will be the users historical articles combined with news relating info
    history_df_cols = [ "user_id","date","history"]
    candidate_df_cols = [ "user_id","date","impressions"]

    history_df = b_df[history_df_cols].copy(deep=True)
    candidate_df = b_df[candidate_df_cols].copy(deep=True)

    # split data into rows
    candidate_df = create_rows(candidate_df, "impressions")
    history_df = create_rows(history_df, "history")
    history_df = history_df.rename(columns={"history":"news_id"}) # rename history column for joining to news df

    # MAKE LABEL 
    candidate_df[["news_id","label"]] =candidate_df["impressions"].str.split(pat="-", expand=True)

    # join news data
    test_can = pd.merge(candidate_df,news_df, on ="news_id").drop(columns=
                                                                    ["impressions","abstract","url","title_entities",	"abstract_entitites"])
    test_history= pd.merge(history_df,news_df, on ="news_id").drop(columns=
                                                                    ["abstract","url","title_entities",	"abstract_entitites"])
    
    #test_can = candidate_df.loc[candidate_df["date"].astype(str) == "2019-11-14"]
    #test_history = history_df.loc[history_df["date"].astype(str) == "2019-11-14"]

    #remove too many users

    can_cnt = test_can.groupby(["user_id"]).size().reset_index(name='counts')
    hist_cnt = test_history.groupby(["user_id"]).size().reset_index(name='counts')
    few_cans = can_cnt.loc[(can_cnt["counts"]< 25) & (can_cnt["counts"]> 5)]
    few_hist = hist_cnt.loc[hist_cnt["counts"]< 25]
    common_users = set(few_cans["user_id"].to_list()).intersection(set(few_hist["user_id"].to_list()))
    common_users = list(common_users)

    hist_sample = test_history.loc[test_history["user_id"].isin(common_users)] # replaced users
    can_sample = test_can.loc[test_can["user_id"].isin(common_users)] # replaced users
    print(hist_sample.shape, can_sample.shape)

    return hist_sample, can_sample

def load_csv(file):
    '''
    Loads a csv to pd df
    '''
    df = pd.read_table(file,sep=",",index_col=0)

    return df

def all_preprocessing_web_app():
    '''
    All the pre-processing needed for the web application
    '''
    # load data
    b_df = load_csv("./data/web_app_users.csv")
    news_df = load_csv("./data/web_app_news.csv")
    news_df_original = news_df.copy(deep=True)

    b_df["date"] = pd.to_datetime(b_df["time"]).dt.date

    # reduce date to only the last date
    #b_df = b_df.loc[b_df["date"].astype(str) == "2019-11-14"]

    # one hot encoding
    # update category and subcategory
    news_cats = news_df.groupby(["category"])["category"].count()
    cats = list(news_cats.index)

    subcats = news_df.groupby(["sub_category"])["sub_category"].count()
    subcats = list(subcats.index)

    one_hot(news_df, "category",cats)
    one_hot(news_df, "sub_category",subcats)

    # filter out multiple impressions per day
    b_df['RN'] = b_df.sort_values(['time'], ascending=[False]).groupby(['user_id']).cumcount() + 1
    b_df= b_df.loc[b_df["RN"] == 1] 


    # make history and cand data
    # History Will be the users historical articles combined with news relating info
    history_df_cols = [ "user_id","date","history"]
    candidate_df_cols = [ "user_id","date","impressions"]

    history_df = b_df[history_df_cols].copy(deep=True)
    candidate_df = b_df[candidate_df_cols].copy(deep=True)

    # split data into rows
    candidate_df = create_rows(candidate_df, "impressions")
    history_df = create_rows(history_df, "history")
    history_df = history_df.rename(columns={"history":"news_id"}) # rename history column for joining to news df

    # MAKE LABEL 
    candidate_df[["news_id","label"]] =candidate_df["impressions"].str.split(pat="-", expand=True)

    # join news data
    candidate_df = pd.merge(candidate_df,news_df, on ="news_id").drop(columns=
                                                                    ["impressions","abstract","url","title_entities",	"abstract_entitites"])
    history_df= pd.merge(history_df,news_df, on ="news_id").drop(columns=
                                                                    ["abstract","url","title_entities",	"abstract_entitites"])

    return history_df, candidate_df,news_df_original

def get_models():
    '''
    Gets the models needed for the NLP experiments
    '''
    model_sts = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    nli_model = CrossEncoder('cross-encoder/nli-roberta-base')
    return model_sts, nli_model

def get_sts():
    '''
    Returns only the STS model
    '''
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


if __name__ == "__main__":
    hist, can = all_preprocessing()
