import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from scipy.stats import linregress


def trend_plot(obj, target_metric, days=7, surgery_date=None, aggregate='mean', linear_trend=False, limit=None):
    period = '%dD' % days
    # Convert the file name into a date object
    figures = []
    for cluster_i in range(obj.n_clusters):
        select = np.where(obj.clusters == cluster_i)[0]
        sub_speed = obj.metrics[target_metric][select]
        sub_date = [datetime.fromtimestamp(bf) for bf in obj.all_timestamp[select]]
        df = pd.DataFrame(data=sub_speed, index=sub_date)

        # Aggregate the measurements by the predefined number of days (period)
        first_day = datetime.fromtimestamp(obj.all_timestamp.min())
        first_day = datetime(first_day.year, first_day.month, first_day.day)
        last_day = datetime.fromtimestamp(obj.all_timestamp.max())
        last_day = datetime(last_day.year, last_day.month, last_day.day)

        if surgery_date:
            # Start the interval on the same day of the week as the surgery day
            offset = first_day.weekday() - surgery_date.weekday()
            n_rep = np.ceil((surgery_date - first_day).days / days)
            before_surgery = pd.date_range(start=first_day + timedelta(offset), periods=n_rep, freq=period,
                                           closed='left')
            n_rep = np.ceil((last_day - surgery_date).days / days) + 1
            after_surgery = pd.date_range(start=surgery_date, periods=n_rep, freq=period)
            dates = before_surgery.union(after_surgery)
        else:
            dates = pd.date_range(start=first_day, end=last_day, freq=period)

        dates_ind = dates.searchsorted(df.index, side='right') - 1
        grouped = df.groupby(dates_ind)
        if aggregate == 'mean':
            df_mean = grouped.mean()
        elif aggregate == 'sum':
            df_mean = grouped.sum()
        else:
            raise Exception('Aggregate mode {} not understood'.format(aggregate))

        fig = plt.figure(figsize=(5, 3))
        figures.append(fig)
        plt.errorbar(df_mean.index + 0.5, df_mean.get_values(), yerr=grouped.std().get_values())

        print('Mean variance: {}'.format(np.nanmean(grouped.std().get_values())))

        if surgery_date:
            surgery_index = dates.searchsorted(surgery_date)
            plt.axvline(surgery_index, linewidth=2, color='k', label='Surgery day')
        else:
            surgery_index = 0

        plt.plot(df_mean.index + 0.5, df_mean.get_values(), 'o', color='orange', zorder=10)

        if linear_trend:
            x_lin = df_mean.index[df_mean.index >= surgery_index] + 0.5
            y_lin = df_mean.get_values()[df_mean.index >= surgery_index, 0]
            not_nan = np.where(np.isfinite(y_lin))[0]
            z = np.poly1d(np.polyfit(x_lin[not_nan], y_lin[not_nan], 1))
            wpm = 365/7/12  # Weeks per month
            plt.plot(x_lin, z(x_lin), '--r', label='Trend: %+.2E m/s/month' % (z[1] * wpm))
            plt.legend()
            slope, intercept, r_value, p_value, std_err = linregress(x_lin, y_lin)
            print('R square trend: {}'.format(r_value))
        else:
            x_lin = df_mean.index + 0.5
            y_lin = df_mean.get_values()[:, 0]
            slope, intercept, r_value, p_value, std_err = linregress(x_lin, y_lin)
            print('R square trend: {}'.format(r_value))


        plt.title('{} - Cluster {}'.format(target_metric, cluster_i))
        plt.xlabel('Week number')

        if limit:
            plt.ylim(limit)

    return figures

