import pandas as pd
import numpy as np


def undercarriage_pad_to_int(l):
    try:
        if l == np.nan or l == 'None or Unspecified':
            return np.nan
        elif type(l) == float:
            return l
        else:
            pad_length = l.split()
            inches = np.float(pad_length[0])
            return inches
    except:
        return l


def tire_size_to_float(l):
    try:
        if l == np.nan or l == 'None or Unspecified':
            return np.nan

        elif l[-1] == '"':
            size = l[:-1]
            return(float(size))
        elif 'inch' in l:
            return float(l[:2])
        else:
            return float(l)
    except:
        return l


def stick_length_to_float(l):
    try:
        if l == np.nan or l == 'None or Unspecified':
            return np.nan
        else:
            length = l.split()
            feet = float(length[0][:-1])
            inches = int(length[1][:-1])
            stick_length = feet + inches/12
            return stick_length
    except:
        return l


def clean_data(n=10):

    data = {'Train': 'Train.zip', 'Test': 'Test.zip'}
    result = {}
    for datatype, path in data.items():

        df = pd.read_csv(path)

        # Replace certain values with NaNs
        df['MachineHoursCurrentMeter'] = df['MachineHoursCurrentMeter'].apply(lambda x: np.nan if x == 0.0 else x)
        df['state'] = df['state'].apply(lambda x: np.nan if x == 'Unspecified' else x)
        df['Transmission'] = df['Transmission'].apply(lambda x: 'Autoshift' if x == 'AutoShift' else x)

        # Convert size columns to floats
        df['Undercarriage_Pad_Width'] = df['Undercarriage_Pad_Width'].apply(undercarriage_pad_to_int)
        df['Tire_Size'] = df['Tire_Size'].apply(tire_size_to_float)
        df['Stick_Length'] = df['Stick_Length'].apply(stick_length_to_float)

        # toss all but the top 10
        for col in ['fiBaseModel', 'fiSecondaryDesc', 'fiModelSeries', 
                    'fiModelDescriptor']:
            lst = list(df[col].value_counts().index[:n])
            df[col] = df[col].apply(lambda x: x if x in lst else np.nan)

        # Get dummies
        df = pd.get_dummies(df, columns=['datasource',
                                         'auctioneerID',
                                         'UsageBand',
                                         'ProductSize',
                                         'state',
                                         'ProductGroup',
                                         'Drive_System',
                                         'Enclosure',
                                         'Forks',
                                         'Pad_Type',
                                         'Ride_Control',
                                         'Stick',
                                         'Transmission',
                                         'Turbocharged',
                                         'Blade_Extension',
                                         'Blade_Width',
                                         'Hydraulics',
                                         'Enclosure_Type',
                                         'Engine_Horsepower',
                                         'Pushblock',
                                         'Ripper',
                                         'Scarifier',
                                         'Tip_Control',
                                         'Coupler',
                                         'Coupler_System',
                                         'Grouser_Tracks',
                                         'Hydraulics_Flow',
                                         'Track_Type',
                                         'Thumb',
                                         'Pattern_Changer',
                                         'Grouser_Type',
                                         'Backhoe_Mounting',
                                         'Blade_Type',
                                         'Travel_Controls',
                                         'Differential_Type',
                                         'Steering_Controls',
                                         'fiBaseModel',
                                         'fiSecondaryDesc',
                                         'fiModelSeries',
                                         'fiModelDescriptor'])

        # Convert to unix time
        df['saledate'] = df['saledate'].astype(np.datetime64).astype(np.int64)

        # Drop columns which give no additional information
        df = df.drop(columns=['MachineID', 'fiModelDesc',
                              'fiProductClassDesc', 'ProductGroupDesc'])

        # Drop columns
        df = df.drop(columns=['Stick_Length', 'Undercarriage_Pad_Width',
                              'Tire_Size', 'saledate',
                              'MachineHoursCurrentMeter'])

        result[datatype] = df

    df_train = result['Train']
    df_test = result['Test']

    # drop any columns with missing data
    df_train = df_train.dropna(axis=1)
    df_test = df_test.dropna(axis=1)

    y_train = np.array(df_train.pop('SalePrice'))

    for col in list(df_train.columns.values):
        if col not in df_test.columns:
            df_train = df_train.drop(columns=[col])

    for col in list(df_test.columns.values):
        if col not in df_train.columns:
            df_test = df_test.drop(columns=[col])

    X_train = np.array(df_train)
    X_test = np.array(df_test)

    return X_train, y_train, df_train, X_test, df_test


if __name__ == '__main__':
    x_train, y_train, df_train, X_test, df_test = clean_data()
