import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def grouped_data(df):
    df_rdd = df.rdd
    df_key_value = (df_rdd.map(lambda x: (x.state,
                               (x.candidate,
                                x.party,
                                x.votes,
                                x.fraction_votes)))
                                )

    ret_df = df_key_value.groupByKey()
    return ret_df

def model(x):
    party_mapping = {'Democrat':0, 'Republican':1}
    column_names = ['candidate', 'party', 'votes', 'fraction_votes']
    df = pd.DataFrame(x, columns=column_names)
    dummpy_vars = pd.get_dummies(df.candidate)
    df2 = pd.concat([dummpy_vars, df], axis=1)
    y = df2.party.apply(lambda x: party_mapping[x]).values
    X = df2.drop(['party', 'candidate'], axis=1).values
    clf = LogisticRegression(random_state=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc

def model_output(grp_df):
    all_outputs = (grp_df.map(lambda x: (x[0], model(list(x[1]))
                            )
                 )
                )
    return all_outputs
