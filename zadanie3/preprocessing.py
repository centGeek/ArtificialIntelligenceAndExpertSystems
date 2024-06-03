from sklearn.preprocessing import LabelEncoder 
import pandas as pd
def processingData():
    file_path_por = './data/student-por.csv'
    df = pd.read_csv(file_path_por)
    df = df['school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3'].str.split(';', expand=True)

    df.columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 
                'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    print(df.info())
    print(df.head())


    df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)

    label_encoder = LabelEncoder()
    # binary_columns = ['sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    # for col in binary_columns:
    #     df[col] = label_encoder.fit_transform(df[col])

    categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df





