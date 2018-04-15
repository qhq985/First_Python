import urllib.request
import re, os, warnings
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import numpy as np
from sklearn import neighbors, model_selection
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import seaborn as sns
from scipy import stats

# Plot style used in this code challenge.
style.use('ggplot')
warnings.filterwarnings('ignore')


# I defined data website outside in case that data website is changed.
# Your browser's user agent is also defined outside the function so you can change it as you want.
# Default one is user agent of Chrome
website = 'http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml'
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ' \
             '(KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36'


# Question 1:
# (1). Programmatically download data from website
def data_download(url, agent=user_agent):
    # Open website in Python, and use regular method to find 2015-09 green taxi data.
    req = urllib.request.Request(url, headers={'User-Agent': agent})
    html = urllib.request.urlopen(req).read().decode('UTF-8')
    p = r'a.*?href="(.*?2015-09.csv)">Green'
    taxi_data_url = re.findall(p, html)[0]

    # Download data set and save as .csv in local folder
    taxi_req = urllib.request.Request(taxi_data_url, headers={'User-Agent': agent})
    data = urllib.request.urlopen(taxi_req).read().decode('UTF-8')
    with open('taxi_data_2015-09.csv', 'w') as f:
        f.write(data)


# If there isn't data set, we will downloan it with our function, otherwise we will skip.
def download_or_reload():
    if not os.path.exists('taxi_data_2015-09.csv'):
        data_download(website)

    # Read data from csv through pandas. And report row number and column number.
    data = pd.read_csv('taxi_data_2015-09.csv')
    row_num = len(data)
    col_num = len(Counter(data.columns))
    print('\nThe columns of data set: {}\n'.format(col_num))
    print('\nThe rows of data set: {}\n'.format(row_num))
    return data



def data_preprocess(data):
    # Describe data to see outliers and NaN.
    description = data.describe()

    # Exclude Egail_fee since it has lots of NaN.
    data.drop(['Ehail_fee'], 1, inplace=True)

    # Change str time into float time.
    def datetime_transform(time, time_type='%Y-%m-%d %H:%M:%S'):
        return dt.datetime.strptime(time, time_type)

    data['pickup_datetime'] = data['lpep_pickup_datetime'].apply(datetime_transform)
    data['dropoff_datetime'] = data['Lpep_dropoff_datetime'].apply(datetime_transform)

    # Calculate pickup hour and dropoff hour.
    data['pickup_hour'] = data['pickup_datetime'].apply(lambda x: x.hour)
    data['dropoff_hour'] = data['dropoff_datetime'].apply(lambda x: x.hour)

    # Drop original str datetime.
    data.drop(['lpep_pickup_datetime', 'Lpep_dropoff_datetime'], 1, inplace=True)

    # Calculate time difference between pick up and drop off..
    data['Timedelta'] = data['dropoff_datetime'] - data['pickup_datetime']
    data['Trip_hours'] = data['Timedelta'].apply(lambda x: x.total_seconds() / 3600)
    data.drop('Timedelta', 1, inplace=True)

    # Calculate average speed fr each trip. And taking the speed limitation into consideration.
    data['Average_speed'] = data['Trip_distance'] / data['Trip_hours']

    # Use isocalendar to derive a new veriable 'week_num'.
    first_week = dt.datetime(2015, 9, 1).isocalendar()[1]

    # Function to be applied in DataFrame.
    def change_to_week(x):
        return x.isocalendar()[1] - first_week + 1

    # Create data for variable 'week_um'.
    data['week_num'] = data['pickup_datetime'].apply(change_to_week)


# Question 2:
# (1). Plot histograms of the number of the trip distance.
def histogram_plot(data):
    # Normally plot without data cleaning.
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    trip_distance = np.array(data['Trip_distance'])
    sns.distplot(trip_distance, bins=50, kde=False)
    ax1.set_title('Hsitogram of Trip Distance')
    ax1.set_xlabel('Trip Distacne (Miles)')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')

    # (2). Remove the outliers and re-plot, and fit it with lognorm.
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    # We exclude outliers which is larger than 3 standard deviation compared to mean.
    # Because data larger than 3 standard deviation can be considerd as impossible.
    std = np.std(trip_distance)
    mu = np.mean(trip_distance)
    # Create a temp data set without outliers
    temp = [i for i in trip_distance if (np.abs(i-mu) < 3*std)]
    sns.distplot(temp, bins=50, kde=False, fit=stats.lognorm)
    ax2.set_title('Hsitogram of Trip Distance (No Outliers, Lognormal)')
    ax2.set_xlabel('Trip Distance (Miles)')
    ax2.set_ylabel('Frequency')
    plt.savefig('Question 2.jpg', format='jpg')
    plt.show()


# Question 3:
# (1). Data manipulation, change strptime (both pick-up time and drop-off time) into float time.
def mean_median_by_hour(data):
    # Use pivot_table method from pandas to create a table which contains mean and median grouped by hour.
    mean_median = data.pivot_table(index='pickup_hour', values='Trip_distance', aggfunc=('mean', 'median'))
    print('Mean and median grouped by hours:\n\n', mean_median)

    # Visualize the mean and median for report.
    plt.plot(mean_median)
    plt.legend(['Mean', 'Median'])
    plt.title('Mean and Median of Trip Distance by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Miles')
    plt.savefig('Question 3-1.jpg', format='jpg')
    plt.show()
    return mean_median


# (2) Trips that are originate or terminate at one of NYC airports (Newark was chosen here).
# By searching the google map manually, I found the rough range of the airport's
# longitude is [-74.19, -74.17], the range of latitude is [40.68, 40.70]. You can
# assume that all the Terminates and parking lots are included.

def newark_airport_data(data):
    # Select data based on longitude range with both pick up and drop off.
    airport_data_longitude = data[((data['Pickup_longitude'] <= -74.17)
                                & (data['Pickup_longitude'] >= -74.19))
                                | ((data['Dropoff_longitude'] <= -74.17)
                                & (data['Dropoff_longitude'] >= -74.19))]

    # Select data based on latitude range with both pick up and drop off.
    airport_data = airport_data_longitude[((airport_data_longitude['Pickup_latitude'] <= 40.70)
                        & (airport_data_longitude['Pickup_latitude'] >= 40.68))
                        | ((airport_data_longitude['Dropoff_latitude'] <= 40.70)
                        & (airport_data_longitude['Dropoff_latitude'] >= 40.68))]

    # Remove trip distance outliers in whole data set.
    std = np.std(data['Trip_distance'])
    mu = np.mean(data['Trip_distance'])
    data = data[(data['Trip_distance'] - mu) < 3 * std]

    # Remove outliers if it's in Newark data set.
    std_airport = np.std(airport_data['Trip_distance'])
    average_airport = np.mean(airport_data['Trip_distance'])
    airport_data = airport_data[(airport_data['Trip_distance'] - average_airport) < 3 * std_airport]

    # Number of trips and average fare.
    num_trips = len(airport_data)

    # Average fare amount excluding outliers.
    average_fare = round(np.mean(airport_data['Fare_amount']), 2)
    average_fare_all = round(np.mean(taxi_data['Fare_amount']), 2)
    # Average distance excluding outliers.
    average_dist = round(np.mean(airport_data['Trip_distance']), 2)
    average_dist_all = round(np.mean(data['Trip_distance']), 2)
    # Average tip amountexcluding outliers.
    average_tip = round(np.mean(airport_data['Tip_amount']),2 )
    average_tip_all = round(np.mean(data['Tip_amount']), 2)

    temp_data = {'Newark':[average_fare, average_dist, average_tip],
                 'All': [average_fare_all, average_dist_all, average_tip_all],
                 'Index': ['Average Fare', 'Average Distance', 'Average Tip']}
    df = pd.DataFrame(temp_data)
    df.set_index('Index')

    # Print number of trips and average statistics.
    print('\nThe number of trips to/from Newark Airport: {0}\n'.format(num_trips))
    print('\nAverage tip to/from Newark, trip distances and tips for all\n:', df)

    ax1 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)

    # Plot tips and average tips over trip distance.
    ax1.scatter(airport_data['Trip_distance'], airport_data['Tip_amount'], s=25, alpha=0.4)
    ax1.plot(airport_data['Trip_distance'], [average_tip]*num_trips, color='b')
    ax1.plot(airport_data['Trip_distance'], [average_tip_all] * num_trips, color='k')
    ax1.set_title('Tip Amount and Average Tips over Trip Distance')
    ax1.set_xlabel('Trip Distance')
    ax1.set_ylabel('Tips Amount')
    ax1.legend(['Average Newark ({})'.format(average_tip),
                'Average All ({})'.format(average_tip_all), 'Tips Amount'])

    # Plot tips and average tips over fare amount.
    ax2.scatter(airport_data['Fare_amount'], airport_data['Tip_amount'], s=25, alpha=0.4)
    ax2.plot(airport_data['Fare_amount'], [average_tip]*num_trips, color='b')
    ax2.plot(airport_data['Fare_amount'], [average_tip_all] * num_trips, color='k')
    ax2.set_title('Tip Amount and Average Tips over Fare Amount')
    ax2.set_xlabel('Fare Amount')
    ax2.set_ylabel('Tips Amount')
    ax2.legend(['Average Newark ({})'.format(average_tip),
                'Average All ({})'.format(average_tip_all), 'Tips Amount'])
    plt.legend()

    plt.savefig('Question 3-2-1.jpg', format='jpg')
    plt.show()
    plt.close()

    # Calculate average tips by hour in whole data set and Newark dataset.
    tips_by_hour_all = data.pivot_table(index='pickup_hour', values='Tip_amount', aggfunc='mean')
    tips_by_hour_newark = airport_data.pivot_table(index='pickup_hour', values='Tip_amount', aggfunc='mean')
    tips_by_hour_all.rename(columns={'Tip_amount': 'Average_tips_all'}, inplace=True)
    tips_by_hour_newark.rename(columns={'Tip_amount': 'Average_tips_newark'}, inplace=True)

    # Reset index to merge to tips data set for ploting purpose, and merge them by inner method.
    tips_by_hour_all.reset_index(inplace=True)
    tips_by_hour_newark.reset_index(inplace=True)
    tips = tips_by_hour_all.merge(tips_by_hour_newark, how='inner')
    tips.set_index('pickup_hour', inplace=True)

    # Visualize average distance, tips and fare. Compared Newark to all.
    labels = ['Average Fare', 'Average Distance', 'Average Tip']
    plt.bar(df.index, df['All'], width=0.1, tick_label=labels)
    plt.bar(df.index+0.1, df['Newark'], width=0.1,
            facecolor = 'lightskyblue', tick_label=labels)
    plt.legend(['All', 'Newark'])
    plt.ylabel('Average')
    plt.xlabel('Items')
    plt.title('Averages over Different Data Set')

    # Add text to bar plot.
    for x, y in zip(df.index, df['All']):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')
    for x, y in zip(df.index+0.1, df['Newark']):
        plt.text(x, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    plt.savefig('Question 3-2-2.jpg', format='jpg')
    plt.legend()
    plt.show()
    

# Question 4:
# Manipulate data, choose information which payment method is credit card. From data dictionary, I know that
# tips data are collected only when paid in credit card.

def tips_PCT(data):
    # Manually choose data for prediction purpose.
    temp = data[['Trip_distance', 'Passenger_count', 'RateCodeID', 'Pickup_longitude', 'Pickup_latitude',
                 'Dropoff_longitude', 'Dropoff_latitude', 'Tip_amount', 'Total_amount', 'Payment_type',
                 'pickup_hour', 'dropoff_hour', 'Trip_type ']]

    # Drop data which total amount is less than 2.5 (initial fee), and RateCodeID is equal to 99 (not in dictionary).
    temp = temp[(temp['Payment_type'] == 1) & (temp['Total_amount'] >= 2.5) & (temp['RateCodeID'] != 99)]

    # Calculate tip percentage of total amount of fare.
    temp['Tip_PCT'] = temp['Tip_amount'] / temp['Total_amount'] * 100
    temp.drop(['Payment_type', 'Tip_amount'], 1, inplace=True)

    # Visualize tip percentage.
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.hist(temp['Tip_PCT'], bins=30, alpha=0.5)
    ax1.set_title('Tip PCT of Total Fare')
    ax1.set_xlabel('Percentages')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')

    plt.savefig('Question 4-1.jpg', format='jpg')
    plt.show()
    
    return temp


# Predictive mdoel, here I predict a range of tip perentage instead of exact percentage.
def predictive_model(data, PCT_range=20):

    # Create label for classification purpose. Tip percentage in rang 0 - 19% will be level 0, 20 - 39% will be
    # level 1, and so on.
    data['Tip_PCT_label'] = np.abs(data['Tip_PCT'] // PCT_range)

    # Choose variables for model.
    target_variable = 'Tip_PCT_label'
    predictors = ['Trip_distance', 'RateCodeID', 'Total_amount', 'pickup_hour',
                  'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude']

    # Create dependent variable and independent variable data set.
    X = np.array(data[predictors])
    y = np.array(data[target_variable])

    # Split data set for traning model and test data.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

    # Model training and testing.
    clf = neighbors.KNeighborsClassifier(n_jobs=8)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test) * 100
    print('\nThe accuracy of prediction (in percentage) when span is {0}: {1}\n'.format(PCT_range, accuracy))


# Question 5:
# Calculate average speed of whole data set.
def average_speed(data):
    # Drop the speed that is larger than 100 and equal 0.
    temp = data[(data['Average_speed'] < 100) & (data['Average_speed'] != 0)].copy()

    # Create average speed table grouped by week.
    speed_table = temp.pivot_table(index='week_num', values='Average_speed', aggfunc='mean')
    print('\nAverage speed by week:\n', speed_table, '\n')

    # Plot histogram of average speed of each trip.
    ax1 = plt.subplot2grid((1, 1), (0, 0))
    sns.distplot(np.array(temp['Average_speed']), bins=50, kde=False)
    ax1.set_title('Average Speed of Each Trip')
    ax1.set_xlabel('Speed')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    plt.savefig('Question 5-1.jpg', format='jpg')
    plt.show()
    plt.close()

    # Plot average speed by week.
    plt.plot(np.array(speed_table))
    plt.title('Average Speed Grouped by Week')
    plt.xlabel('Week')
    plt.ylabel('Average Speed')

    plt.savefig('Question 5-2.jpg', format='jpg')
    plt.show()

    # Return the data that is needed in ANOVA analysis.
    return temp[['pickup_hour', 'Average_speed', 'week_num']]


# Conduct analysis of variance to detect whether there is difference in average speed.
def anova_test(data):
    # ANOVA for average speed against week number.
    model_week = ols('Average_speed ~ week_num', data).fit()
    anova_week = anova_lm(model_week)

    # Anova for average speed against hours.
    model_hour = ols('Average_speed ~ pickup_hour', data).fit()
    anova_hour = anova_lm(model_hour)

    print('\nANOVA test on average speed against week:\n', anova_week)
    print('\nANOVA test on average speed against hour:\n', anova_hour)

    # Visualize data with boxplot.
    sns.boxplot(x='pickup_hour', y='Average_speed', data=data)
    plt.title('Average Speed by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Average Speed')
    plt.savefig('Question 5-3.jpg', format='jpg')
    plt.show()
    plt.close()

    sns.boxplot(x='week_num', y='Average_speed', data=data)
    plt.title('Average Speed by week')
    plt.xlabel('Week')
    plt.ylabel('Average Speed')
    plt.savefig('Question 5-4.jpg', format='jpg')
    plt.show()
    plt.close()

# Call the function to download for further analysis.
taxi_data = download_or_reload()

# Data preprocessing.
data_preprocess(taxi_data)

# Call plot function, plot histogram and save image.
histogram_plot(taxi_data)

# Call function to calculate mean and median groupd by hour, and visualize it.
mean_median_table = mean_median_by_hour(taxi_data)


# Return average fare of trips to/from Newark, and will visualize some ohter
# characteristics that I am interested in.
newark_airport_data(taxi_data)

# Return a data set for model training and testing.
model_data = tips_PCT(taxi_data)


# Print out the accuracy of model.
predictive_model(model_data)


# Return a data set for ANOVA test. And visualize average speed data by week and hour.
anova_data = average_speed(taxi_data)


# ANOVA test and boxplot of average speed.
anova_test(anova_data)
