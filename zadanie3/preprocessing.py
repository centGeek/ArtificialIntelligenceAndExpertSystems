from sklearn.preprocessing import LabelEncoder 
import pandas as pd
def processingData(file_path):
    df = pd.read_csv(file_path)
    df = df['school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;guardian;traveltime;studytime;failures;schoolsup;famsup;paid;activities;nursery;higher;internet;romantic;famrel;freetime;goout;Dalc;Walc;health;absences;G1;G2;G3'].str.split(';', expand=True)

    df.columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 
                'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 
                'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

    df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)

    label_encoder = LabelEncoder()

    categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                        'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df

def assign_grade(value):
    if value <= 6:
        return 'bardzo niska'
    if value <= 10:
        return 'niska'
    elif value <= 14:
        return 'średnia'
    elif value <=17:
        return 'wysoka'
    else:
        return 'bardzo wysoka'





